# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-many-instance-attributes

import sympy as sp
from hcl_mlir.exceptions import (
    HCLError,
    APIError,
    TensorError,
    HCLNotImplementedError,
    HCLValueError,
    DTypeError,
    DTypeWarning,
)

from .registry import get_type_rules, register_type_rules
from .type_rules import (
    add_sub_rule,
    mul_rule,
    mod_rule,
    and_or_rule,
    logic_op_rule,
    cmp_rule,
    div_rule,
    pow_rule,
    shift_rule,
    select_rule,
    intrin_rule,
)
from ..types import Int, UInt, Index, Float, Struct, dtype_to_str


def print_indent(string, level):
    for _ in range(level):
        string += "  "
    return string


def immediate_to_constant(value, loc, dtype=None):
    if not isinstance(value, (int, float)):
        return value  # pass through
    if dtype is not None:
        return ConstantOp(value, dtype, loc)
    if isinstance(value, bool):
        return ConstantOp(value, Int(1), loc)
    if isinstance(value, int):
        return ConstantOp(
            value, Int(32 if value < 0xFFFFFFFF else value.bit_length()), loc
        )
    return ConstantOp(value, Float(64), loc)


def replace_all_uses_with(op, old_tensor, new_tensor):
    if isinstance(op, FuncOp):
        new_rets = []
        for ret in op.return_tensors:
            if ret.name == old_tensor.name:
                new_rets.append(new_tensor)
            else:
                new_rets.append(ret)
        op.return_tensors = new_rets

    for attr, value in op.__dict__.items():
        if attr == "tensor" and value.name == old_tensor.name:
            setattr(op, attr, new_tensor)
        if isinstance(value, list):
            for v in value:
                replace_all_uses_with(v, old_tensor, new_tensor)
        if hasattr(value, "__dict__"):
            replace_all_uses_with(value, old_tensor, new_tensor)


# Unwrap sympy integer or float into python integer or float
def unwrap_sp(expr):
    if isinstance(expr, sp.core.numbers.Integer):
        return int(expr)
    if isinstance(expr, sp.core.numbers.Float):
        return float(expr)
    return expr


def simplify(expr):
    # pylint: disable=too-many-return-statements, too-many-branches
    """
    simplifies an expression by replacing all constants with their values
    and compute the result if possible
    Only supports affine expressions on integers and floats
    """
    if isinstance(expr, (int, float)):
        return expr
    if isinstance(expr, sp.core.numbers.Integer):
        return int(expr)
    if isinstance(expr, ConstantOp):
        return expr.value
    if isinstance(expr, IterVar):
        return sp.symbols(expr.name)
    if isinstance(expr, Add):
        return sp.simplify(simplify(expr.lhs) + simplify(expr.rhs))
    if isinstance(expr, Sub):
        return sp.simplify(simplify(expr.lhs) - simplify(expr.rhs))
    if isinstance(expr, Mul):
        return sp.simplify(simplify(expr.lhs) * simplify(expr.rhs))
    if isinstance(expr, Div):
        return sp.simplify(simplify(expr.lhs) / simplify(expr.rhs))
    if isinstance(expr, FloorDiv):
        return sp.simplify(simplify(expr.lhs) // simplify(expr.rhs))
    if isinstance(expr, Mod):
        return sp.simplify(simplify(expr.lhs) % simplify(expr.rhs))
    if isinstance(expr, LoadOp):
        tensor = expr.tensor
        if tensor.fcompute is None:
            return expr
        index = expr.index
        return sp.simplify(simplify(tensor.fcompute(*index)))
    if isinstance(expr, LeftShiftOp):
        lhs = unwrap_sp(simplify(expr.lhs))
        rhs = unwrap_sp(simplify(expr.rhs))
        return sp.simplify(lhs << rhs)
    if isinstance(expr, RightShiftOp):
        lhs = unwrap_sp(simplify(expr.lhs))
        rhs = unwrap_sp(simplify(expr.rhs))
        return sp.simplify(lhs >> rhs)
    if isinstance(expr, And):
        lhs = unwrap_sp(simplify(expr.lhs))
        rhs = unwrap_sp(simplify(expr.rhs))
        return sp.simplify(lhs & rhs)
    if isinstance(expr, Or):
        lhs = unwrap_sp(simplify(expr.lhs))
        rhs = unwrap_sp(simplify(expr.rhs))
        return sp.simplify(lhs | rhs)
    if isinstance(expr, XOr):
        lhs = unwrap_sp(simplify(expr.lhs))
        rhs = unwrap_sp(simplify(expr.rhs))
        return sp.simplify(lhs ^ rhs)
    if isinstance(expr, CastOp):
        return simplify(expr.expr)
    if isinstance(expr, LogicalAnd):
        lhs = unwrap_sp(simplify(expr.lhs))
        rhs = unwrap_sp(simplify(expr.rhs))
        return sp.simplify(lhs and rhs)
    if isinstance(expr, LogicalOr):
        lhs = unwrap_sp(simplify(expr.lhs))
        rhs = unwrap_sp(simplify(expr.rhs))
        return sp.simplify(lhs or rhs)
    if isinstance(expr, Cmp):
        lhs = unwrap_sp(simplify(expr.lhs))
        rhs = unwrap_sp(simplify(expr.rhs))
        op = expr.name
        if op == "lt":
            output = lhs < rhs
        elif op == "le":
            output = lhs <= rhs
        elif op == "eq":
            output = lhs == rhs
        elif op == "ne":
            output = lhs != rhs
        elif op == "gt":
            output = lhs > rhs
        elif op == "ge":
            output = lhs >= rhs
        else:
            raise HCLError(f"Unsupported expression type: {type(expr)}, {expr.name}")
        if output:
            return sp.simplify(1)
        return sp.simplify(0)
    if isinstance(expr, Neg):
        return sp.simplify(-unwrap_sp(simplify(expr.expr)))
    if isinstance(expr, StructGetOp):
        struct = expr.struct
        index = struct.index
        e = struct.tensor.fcompute(*index)[expr.field]
        return sp.simplify(simplify(e))
    if isinstance(expr, SelectOp):  # pylint: disable=no-else-return
        if simplify(expr.cond):
            return sp.simplify(simplify(expr.true_value))
        return sp.simplify(simplify(expr.false_value))
    else:
        raise HCLError(f"Unsupported expression type: {type(expr)}")


class Location:
    """Filename and linenumber"""

    def __init__(self, filename, lineno):
        self.filename = filename
        self.lineno = lineno

    def __str__(self):
        return f"{self.filename}:{self.lineno}"


class Scope:
    """Scope class to manage operation insertion scopes."""

    def __init__(self):
        self.stack = []

    def push(self, new_scope: list):
        self.stack.append(new_scope)

    def pop(self):
        return self.stack.pop()

    def get(self):
        return self.stack[-1]

    def __repr__(self):
        return str(self.stack)

    def __len__(self):
        return len(self.stack)

    def reset(self):
        self.stack.clear()
        # this list is for operations
        # that are not enclosed in a top-level function
        # in the case that there is a top-level function,
        # this list will be poped and the operations will
        # be inserted into the top-level function body scope
        self.stack.append([])


scope = Scope()
scope.reset()


class Operation:
    """Base class for all operations.

    Parameters
    ----------
    name : str
        The name of the operation

    """

    def __init__(self, name, loc):
        self.name = name
        self.loc = loc
        # the MLIR operation
        self.ir_op = None
        # the MLIR operation's result
        # when an operation is built, its result will be set
        self.result = None
        # whether a new MLIR operation is built
        # when .result is present
        self.reusable = True

    def __repr__(self):
        return self.name


class Expr:
    """Base class for all expressions.

    Parameters
    ----------
    name : str
        The name of the expression

    """

    def __init__(self, name, loc):
        self.name = name
        self.loc = loc
        self.dtype = None
        self.tinf_engine = TypeInference()
        # When an expression is built, its result will be set
        self.result = None
        # whether a new MLIR operation is built
        # when .result is present
        # If reusable is False, a new MLIR operation will be built
        # everytime an expression is used
        self.reusable = False

    def __repr__(self):
        return self.name

    def __add__(self, other):
        return Add(self, other, self.loc)

    def __radd__(self, other):
        return Add(other, self, self.loc)

    def __sub__(self, other):
        return Sub(self, other, self.loc)

    def __rsub__(self, other):
        return Sub(other, self, self.loc)

    def __mul__(self, other):
        return Mul(self, other, self.loc)

    def __rmul__(self, other):
        return Mul(other, self, self.loc)

    def __div__(self, other):
        return Div(self, other, self.loc)

    def __rdiv__(self, other):
        return Div(other, self, self.loc)

    def __truediv__(self, other):
        return Div(self, other, self.loc)

    def __rtruediv__(self, other):
        return Div(other, self, self.loc)

    def __floordiv__(self, other):
        return FloorDiv(self, other, self.loc)

    def __rfloordiv__(self, other):
        return FloorDiv(other, self, self.loc)

    def __mod__(self, other):
        return Mod(self, other, self.loc)

    def __rmod__(self, other):
        return Mod(other, self, self.loc)

    def __neg__(self):
        return Neg(self, self.loc)

    def __lshift__(self, other):
        return LeftShiftOp(self, other, self.loc)

    def __rlshift__(self, other):
        return LeftShiftOp(other, self, self.loc)

    def __rshift__(self, other):
        return RightShiftOp(self, other, self.loc)

    def __rrshift__(self, other):
        return RightShiftOp(other, self, self.loc)

    def __and__(self, other):
        return And(self, other, self.loc)

    def __rand__(self, other):
        return And(other, self, self.loc)

    def __or__(self, other):
        return Or(self, other, self.loc)

    def __ror__(self, other):
        return Or(other, self, self.loc)

    def __xor__(self, other):
        return XOr(self, other, self.loc)

    def __rxor__(self, other):
        return XOr(other, self, self.loc)

    def __invert__(self):
        return Invert(self, self.loc)

    def __lt__(self, other):
        if other is None:
            return False
        return Cmp("lt", self, other, self.loc)

    def __le__(self, other):
        return Cmp("le", self, other, self.loc)

    def __eq__(self, other):
        if other is None:
            return False
        return Cmp("eq", self, other, self.loc)

    def __ne__(self, other):
        if other is None:
            return True
        return Cmp("ne", self, other, self.loc)

    def __gt__(self, other):
        return Cmp("gt", self, other, self.loc)

    def __ge__(self, other):
        return Cmp("ge", self, other, self.loc)

    def __getitem__(self, indices):
        """Bit slicing and bit selection"""
        # pylint: disable=no-else-return
        if isinstance(indices, slice):
            lo, hi = indices.start, indices.stop
            if isinstance(lo, int) and isinstance(hi, int):
                if lo > hi:
                    raise APIError(
                        "Lower bound should be smaller than upper bound. Use `.reverse()` if you want to reverse the bits"
                    )
                if lo == hi:
                    return self
                else:
                    return GetSliceOp(self, lo, hi - 1, self.loc)
            else:
                return GetSliceOp(self, lo, hi - 1, self.loc)
        else:
            if not isinstance(indices, tuple):
                indices = (indices,)
            if not len(indices) == 1:
                raise APIError("Can only access one bit of the integer")
            index = indices[0]
            return GetBitOp(self, index, self.loc)

    def __setitem__(self, indices, expr):
        region = scope.get()
        # pylint: disable=no-else-raise
        if isinstance(indices, slice):
            lo, hi = indices.start, indices.stop
            if isinstance(lo, int) and isinstance(hi, int):
                if lo > hi:
                    raise APIError(
                        "Lower bound should be smaller than upper bound. Use `.reverse()` if you want to reverse the bits"
                    )
                elif lo == hi:  # e.g. [2:2]
                    if not isinstance(expr, LoadOp):
                        raise APIError(
                            "Please check the expression to make sure the lower bound not equal to the upper bound"
                        )
                    else:
                        store_op = StoreOp(expr, self.tensor, self.indices, self.loc)
                        region.append(store_op)
                else:
                    setslice_op = SetSliceOp(self, lo, hi - 1, expr, self.loc)
                    region.append(setslice_op)
            else:
                setslice_op = SetSliceOp(self, lo, hi - 1, expr, self.loc)
                region.append(setslice_op)
        else:
            if not isinstance(indices, tuple):
                indices = (indices,)
            if not len(indices) == 1:
                raise APIError("Can only access one bit of the integer")
            indices = indices[0]
            setbit_op = SetBitOp(self, indices, expr, self.loc)
            region.append(setbit_op)

    def reverse(self):
        return BitReverseOp(self, self.loc)

    def __nonzero__(self):
        raise APIError(
            "1) Cannot use and / or / not operator to Expr, "
            + "2) Cannot compare NumPy numbers with HeteroCL exprs, "
            + "hint: swap the operands"
        )

    def __bool__(self):
        return self.__nonzero__()

    def equal(self, other):
        # TODO(Niansong): not sure when this method is called
        # throw an error for now
        raise HCLNotImplementedError("equal is not implemented yet")

    def astype(self, dtype):
        return CastOp(self, dtype, self.loc)

    def __getattr__(self, key):
        """Access a field of a struct value"""
        # bypass the attribute lookup to avoid infinite recursion
        if key in self.__dict__:
            return self.__dict__[key]
        if isinstance(self, LoadOp):
            # access a field from a struct tensor
            key_list = list(self.tensor.dtype.dtype_dict.keys())
            if key not in key_list:
                raise HCLValueError(
                    f"No such field: {key} in struct {self.tensor.dtype}"
                )
            key_idx = key_list.index(key)
            return StructGetOp(self, key_idx, self.loc)
        # We don't throw an error here
        # because the user may be trying to test if
        # an attribute exists with hasattr().
        return None


class UnaryOp(Expr):
    """Base class for all unary operations.

    Parameters
    ----------
    op : str
        The name of the operation

    """

    def __init__(self, op, expr, loc):
        super().__init__(op, loc)
        expr = immediate_to_constant(expr, loc)
        self.expr = expr

    def __repr__(self):
        return f"({self.name} {self.expr})"


class BinaryOp(Expr):
    """Base class for all binary operations.

    Parameters
    ----------
    op : str
        The name of the operation

    """

    def __init__(self, op, lhs, rhs, loc):
        super().__init__(op, loc)
        lhs = immediate_to_constant(lhs, loc)
        rhs = immediate_to_constant(rhs, loc)
        self.lhs = lhs
        self.rhs = rhs
        self.dtype = self.tinf_engine.infer(self)

    def __repr__(self):
        return f"({self.lhs} {self.name} {self.rhs})"


class TernaryOp(Expr):
    """Base class for all ternary operations.

    Parameters
    ----------
    op : str
        The name of the operation

    """

    def __init__(self, op, cond, lhs, rhs, loc):
        super().__init__(op, loc)
        cond = immediate_to_constant(cond, loc)
        lhs = immediate_to_constant(lhs, loc)
        rhs = immediate_to_constant(rhs, loc)
        self.cond = cond
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"({self.cond} ? {self.lhs} : {self.rhs})"


class CastOp(Expr):
    """Cast an expression to a given type.

    Parameters
    ----------

    """

    def __init__(self, expr, dtype, loc):
        super().__init__(dtype_to_str(dtype), loc)
        expr = immediate_to_constant(expr, loc)
        self.expr = expr
        self.dtype = dtype

    def __repr__(self):
        return f"({self.dtype}) {self.expr}"


@register_type_rules(add_sub_rule)
class Add(BinaryOp):
    """Addition operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__("+", lhs, rhs, loc)


@register_type_rules(add_sub_rule)
class Sub(BinaryOp):
    """Subtraction operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__("-", lhs, rhs, loc)


@register_type_rules(mul_rule)
class Mul(BinaryOp):
    """Multiplication operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__("*", lhs, rhs, loc)


@register_type_rules(div_rule)
class Div(BinaryOp):
    """Division operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__("/", lhs, rhs, loc)


@register_type_rules(select_rule)
class Min(BinaryOp):
    """Min operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__("min", lhs, rhs, loc)

    def __repr__(self):
        return f"min({self.lhs}, {self.rhs})"


@register_type_rules(select_rule)
class Max(BinaryOp):
    """Max operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__("max", lhs, rhs, loc)

    def __repr__(self):
        return f"max({self.lhs}, {self.rhs})"


@register_type_rules(div_rule)
class FloorDiv(BinaryOp):
    """Floor division operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__("//", lhs, rhs, loc)


@register_type_rules(mod_rule)
class Mod(BinaryOp):
    """Modulo operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__("%", lhs, rhs, loc)


@register_type_rules(shift_rule)
class LeftShiftOp(BinaryOp):
    """Left shift operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__("<<", lhs, rhs, loc)


@register_type_rules(shift_rule)
class RightShiftOp(BinaryOp):
    """Right shift operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__(">>", lhs, rhs, loc)


@register_type_rules(cmp_rule)
class Cmp(BinaryOp):
    """Comparison operation."""

    def __init__(self, op, lhs, rhs, loc):
        super().__init__(op, lhs, rhs, loc)
        self.dtype = UInt(1)


@register_type_rules(and_or_rule)
class And(BinaryOp):
    """Bitwise and operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__("&", lhs, rhs, loc)


@register_type_rules(and_or_rule)
class Or(BinaryOp):
    """Bitwise or operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__("|", lhs, rhs, loc)


@register_type_rules(and_or_rule)
class XOr(BinaryOp):
    """Bitwise xor operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__("^", lhs, rhs, loc)


@register_type_rules(and_or_rule)
class Invert(UnaryOp):
    """Bitwise invert operation, e.g. 0b1011 -> 0b0100."""

    def __init__(self, expr, loc):
        super().__init__("~", expr, loc)
        self.dtype = self.tinf_engine.infer(self)


class Neg(UnaryOp):
    """Negate operation, i.e. -x for any expression x."""

    def __init__(self, expr, loc):
        super().__init__("neg", expr, loc)
        self.dtype = self.tinf_engine.infer(self)


class BitReverseOp(UnaryOp):
    """Bit reverse operation."""

    def __init__(self, expr, loc):
        super().__init__("bit_reverse", expr, loc)
        self.dtype = self.tinf_engine.infer(self)


class BitCastOp(UnaryOp):
    """Bit cast operation."""

    def __init__(self, expr, dtype, loc):
        super().__init__("bit_cast", expr, loc)
        self.dtype = dtype


@register_type_rules(intrin_rule)
class MathExpOp(UnaryOp):
    """Mathematical exponential operation."""

    def __init__(self, expr, loc):
        super().__init__("exp", expr, loc)
        self.dtype = self.tinf_engine.infer(self)


@register_type_rules(pow_rule)
class MathPowOp(BinaryOp):
    """Mathematical power operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__("pow", lhs, rhs, loc)
        self.dtype = self.tinf_engine.infer(self)


@register_type_rules(intrin_rule)
class MathLogOp(UnaryOp):
    """Mathematical log operation."""

    def __init__(self, expr, loc):
        super().__init__("log", expr, loc)
        self.dtype = self.tinf_engine.infer(self)


@register_type_rules(intrin_rule)
class MathLog2Op(UnaryOp):
    """Mathematical log2 operation."""

    def __init__(self, expr, loc):
        super().__init__("log2", expr, loc)
        self.dtype = self.tinf_engine.infer(self)


@register_type_rules(intrin_rule)
class MathLog10Op(UnaryOp):
    """Mathematical log10 operation."""

    def __init__(self, expr, loc):
        super().__init__("log10", expr, loc)
        self.dtype = self.tinf_engine.infer(self)


@register_type_rules(intrin_rule)
class MathSqrtOp(UnaryOp):
    """Mathematical square root operation."""

    def __init__(self, expr, loc):
        super().__init__("sqrt", expr, loc)
        self.dtype = self.tinf_engine.infer(self)


@register_type_rules(intrin_rule)
class MathSinOp(UnaryOp):
    """Mathematical sine operation."""

    def __init__(self, expr, loc):
        super().__init__("sin", expr, loc)
        self.dtype = self.tinf_engine.infer(self)


@register_type_rules(intrin_rule)
class MathCosOp(UnaryOp):
    """Mathematical cosine operation."""

    def __init__(self, expr, loc):
        super().__init__("cos", expr, loc)
        self.dtype = self.tinf_engine.infer(self)


@register_type_rules(intrin_rule)
class MathTanOp(UnaryOp):
    """Mathematical tangent operation."""

    def __init__(self, expr, loc):
        super().__init__("tan", expr, loc)
        self.dtype = self.tinf_engine.infer(self)


@register_type_rules(intrin_rule)
class MathTanhOp(UnaryOp):
    """Mathematical hyperbolic tangent operation."""

    def __init__(self, expr, loc):
        super().__init__("tanh", expr, loc)
        self.dtype = self.tinf_engine.infer(self)


@register_type_rules(logic_op_rule)
class LogicalAnd(BinaryOp):
    """Logical and operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__("&&", lhs, rhs, loc)


@register_type_rules(logic_op_rule)
class LogicalOr(BinaryOp):
    """Logical or operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__("||", lhs, rhs, loc)


@register_type_rules(logic_op_rule)
class LogicalXOr(BinaryOp):
    """Logical xor operation."""

    def __init__(self, lhs, rhs, loc):
        super().__init__("^^", lhs, rhs, loc)


class PrintMemRefOp(Operation):
    """Print memref operation."""

    def __init__(self, memref, dtype, loc):
        super().__init__("print_memref", loc)
        self.memref = memref
        self.dtype = dtype
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str += print_indent(code_str, self.level)
        code_str += f"print_memref({self.name} {self.memref})"
        return code_str


class ConstantOp(Expr):
    """Constant scalar operation."""

    def __init__(self, value, dtype, loc):
        super().__init__(str(value), loc)
        self.value = value
        self.dtype = dtype

    def __repr__(self):
        return f"{self.value}"


class ConstantTensorOp(Expr):
    """Constant tensor operation."""

    # TODO(Niansong): handle overflow
    def __init__(self, values, name, shape, dtype, loc):
        super().__init__(name, loc)
        self.values = values
        self.dtype = dtype
        self.shape = shape
        self.tensor = AllocOp(name, shape, dtype, loc)
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str += print_indent(code_str, self.level)
        code_str += f"{self.name} = constant_tensor({self.values.shape}, {self.dtype})"
        return code_str


class LoadOp(Expr):
    """Load operation."""

    def __init__(self, tensor, index, loc):
        super().__init__("getitem", loc)
        self.tensor = tensor
        self.index = [immediate_to_constant(i, loc, Index()) for i in index]
        self.dtype = tensor.dtype
        # load is not reusable
        # e.g.
        # const = hcl.scalar(64).v
        # v1 = hcl.scalar(const + 1)
        # v2 = hcl.scalar(const + 2)
        # each `const` is a load operation,
        # but they are not reusable, meaning
        # that we build two different load operations
        self.reusable = False

    def __repr__(self):
        return f"{self.tensor.name}[" + ", ".join([str(i) for i in self.index]) + "]"


class StoreOp(Operation):
    """Store operation."""

    def __init__(self, tensor, index, value, loc):
        super().__init__("setitem", loc)
        self.tensor = tensor
        self.index = [immediate_to_constant(i, loc, Index()) for i in index]
        self.value = immediate_to_constant(value, loc, tensor.dtype)
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += (
            f"{self.tensor.name}["
            + ", ".join([str(i) for i in self.index])
            + f"] = {self.value}"
        )
        return code_str


class GetBitOp(Expr):
    """Get bit operation"""

    def __init__(self, expr, index, loc):
        super().__init__("getbit", loc)
        self.expr = immediate_to_constant(expr, loc)
        self.index = immediate_to_constant(index, loc, Index())
        self.dtype = UInt(1)

    def __repr__(self):
        return f"{self.expr}[{self.index}]"


class SetBitOp(Operation):
    """Set bit operation"""

    def __init__(self, expr, index, value, loc):
        super().__init__("setbit", loc)
        self.expr = expr
        self.index = immediate_to_constant(index, loc, Index())
        self.value = immediate_to_constant(value, loc, UInt(1))
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"{self.expr}[{self.index}] = {self.value}"
        return code_str


class GetSliceOp(Expr):
    """Get slice operation"""

    def __init__(self, expr, start, end, loc):
        super().__init__("getslice", loc)
        self.expr = expr
        self.start = immediate_to_constant(start, loc, Index())
        self.end = immediate_to_constant(end, loc, Index())
        # using fixed point or float bound is considered unsafe
        # throw error and prompt for explicit cast
        if not isinstance(self.start.dtype, (Int, UInt, Index)):
            raise DTypeError(
                f"Start index {self.start} of {self} must be an integer or index."
                + f"Got {type(self.start)} instead."
                + " Please cast it explicitly with hcl.cast(dtype, expr)."
            )
        if not isinstance(self.end.dtype, (Int, UInt, Index)):
            raise DTypeError(
                f"End index {self.end} of {self} must be an integer or index."
                + f"Got {type(self.end)} instead."
                + " Please cast it explicitly with hcl.cast(dtype, expr)."
            )
        bitwidth = self.end - self.start + 1
        bitwidth = simplify(bitwidth)
        if bitwidth.is_constant():
            self.dtype = UInt(int(bitwidth))
        else:
            self.dtype = None
            DTypeWarning(
                f"{self}'s bitwidth cannot be determined at compile time."
            ).warn()

    def __repr__(self):
        return f"{self.expr}[{self.start}:{self.end}]"


class SetSliceOp(Operation):
    """Set slice operation"""

    def __init__(self, expr, start, end, value, loc):
        super().__init__("setslice", loc)
        self.expr = expr
        self.start = immediate_to_constant(start, loc, Index())
        self.end = immediate_to_constant(end, loc, Index())
        # using fixed point or float bound is considered unsafe
        # throw error and prompt for explicit cast
        if not isinstance(self.start.dtype, (Int, UInt, Index)):
            raise DTypeError(
                f"Start index {self.start} of {self} must be an integer or index."
                + f"Got {type(self.start)} instead."
                + " Please cast it explicitly with hcl.cast(dtype, expr)."
            )
        if not isinstance(self.end.dtype, (Int, UInt, Index)):
            raise DTypeError(
                f"End index {self.end} of {self} must be an integer or index."
                + f"Got {type(self.end)} instead."
                + " Please cast it explicitly with hcl.cast(dtype, expr)."
            )
        self.value = immediate_to_constant(value, loc)
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"{self.expr}[{self.start}:{self.end}] = {self.value}"
        return code_str


class TensorSlice(Expr):
    def __init__(self, full_shape, dtype, parent, indices, loc, name=None):
        super().__init__(name, loc)
        self.full_shape = full_shape
        self.dtype = dtype
        self.name = name
        self.parent = parent
        self.indices = indices
        # calculate tensor slice shape
        shape = []
        dims = 0
        for index in indices:
            if isinstance(index, int):
                dims += 1
            elif isinstance(index, slice):
                step = index.step if index.step is not None else 1
                dim_size = (index.stop - index.start) / step
                shape.append(int(dim_size))
                dims += 1
            # index is an expr
            elif isinstance(index, Expr):
                dims += 1
        for i, dim in enumerate(self.full_shape):
            if i < dims:
                continue
            shape.append(dim)
        self.shape = tuple(shape)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        # check indices dtype
        for index in indices:
            if isinstance(index, (int, slice)):
                continue
            if not isinstance(index.dtype, (UInt, Int, Index)):
                raise DTypeError(
                    f"Index {index} of {self} must be an integer or index."
                    + f"Got {type(index)} instead."
                    + " Please cast it explicitly with hcl.cast(dtype, expr)."
                )
        if len(self.indices + indices) < len(self.full_shape):
            return TensorSlice(
                self.full_shape,
                self.dtype,
                self.parent,
                self.indices + indices,
                self.loc,
                self.name,
            )
        if len(self.indices + indices) == len(self.full_shape):
            # format indices
            new_indices = []
            for index in self.indices + indices:
                index = immediate_to_constant(index, self.loc)
                new_indices.append(index)
            load = LoadOp(self.parent, new_indices, self.loc)
            return load
        raise TensorError("Indices length > # of array dimensions")

    def __setitem__(self, indices, expr):
        if not isinstance(indices, tuple):
            indices = (indices,)
        # check indices dtype
        for index in indices:
            if isinstance(index, (int, slice)):
                continue
            if not isinstance(index.dtype, (UInt, Int, Index)):
                raise DTypeError(
                    f"Index {index} of {self} must be an integer or index."
                    + f"Got {type(index)} instead."
                    + " Please cast it explicitly with hcl.cast(dtype, expr)."
                )
        if len(self.indices + indices) < len(self.full_shape):
            raise HCLNotImplementedError("Writing to a slice of tensor is not allowed.")
        if len(self.indices + indices) == len(self.full_shape):
            new_indices = []
            for index in list(self.indices) + list(indices):
                index = immediate_to_constant(index, self.loc)
                new_indices.append(index)
            expr = immediate_to_constant(expr, self.loc)
            store_op = StoreOp(self.parent, new_indices, expr, self.loc)
            region = scope.get()
            region.append(store_op)
        else:
            raise TensorError(
                "Indices length > # of array dimensions,"
                + f"indices=[{self.indices + indices}], shape={self.full_shape}"
            )


class AllocOp(Expr):
    """Allocate memory for a buffer

    Parameters
    ----------
    name : str
        The name of the operation

    """

    def __init__(self, name, shape, dtype, loc):
        super().__init__(name, loc)
        self.shape = shape
        self.dtype = dtype
        # an optional reference to python function that computes the tensor
        self.fcompute = None
        # uses is a list of ComputeOp that uses the tensor produced by this op
        # we need such list to support create_schedule without an enclosing function
        self.uses = []
        # Axes, a list of loop handles corresponding to the loop axes
        self.axis = []
        # the device where the tensor is allocated
        # e.g. Host, FPGA, GPU
        self.device = None
        self.level = None
        self.reusable = True

    def __repr__(self):
        code_str = ""
        if self.level is not None:
            code_str = print_indent(code_str, self.level)
        code_str += f"{self.name} = alloc({self.shape}, {self.dtype})"
        return code_str

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        # check indices dtype
        for index in indices:
            if isinstance(index, (int, slice)):
                continue
            if not isinstance(index.dtype, (UInt, Int, Index)):
                raise DTypeError(
                    f"Index {index} of {self} must be an integer or index."
                    + f"Got {type(index)} instead."
                    + " Please cast it explicitly with hcl.cast(dtype, expr)."
                )
        # if we are slicing tensor
        if len(indices) < len(self.shape):
            return TensorSlice(
                full_shape=self.shape,
                dtype=self.dtype,
                parent=self,
                indices=indices,
                loc=self.loc,
                name=self.name,
            )
        # if we are loading a value from the tensor
        if len(indices) == len(self.shape):
            # format indices
            new_indices = []
            for index in indices:
                index = immediate_to_constant(index, self.loc)
                new_indices.append(index)
            load = LoadOp(self, new_indices, self.loc)
            return load
        raise TensorError("Indices length > # of array dimensions")

    def __setitem__(self, indices, value):
        if not isinstance(indices, tuple):
            indices = (indices,)
        # check indices dtype
        for index in indices:
            if isinstance(index, (int, slice)):
                continue
            if not isinstance(index.dtype, (UInt, Int, Index)):
                raise DTypeError(
                    f"Index {index} of {self} must be an integer or index."
                    + f"Got {type(index)} instead."
                    + " Please cast it explicitly with hcl.cast(dtype, expr)."
                )
        if len(indices) < len(self.shape):
            raise HCLNotImplementedError("Writing to a slice of tensor is not allowed.")
        if len(indices) == len(self.shape):
            # format indices
            new_indices = []
            for index in indices:
                index = immediate_to_constant(index, self.loc)
                new_indices.append(index)
            expr = immediate_to_constant(value, self.loc)
            store_op = StoreOp(self, new_indices, expr, self.loc)
            # StoreOp is an operation
            # so we need to add it to the current scope
            region = scope.get()
            region.append(store_op)
        else:
            raise TensorError("Indices length > # of array dimensions")

    @property
    def v(self):
        if len(self.shape) == 1 and self.shape[0] == 1:
            return self[0]
        raise TensorError(".v can only be used on scalars")

    @v.setter
    def v(self, value):
        if len(self.shape) == 1 and self.shape[0] == 1:
            value = immediate_to_constant(value, self.loc)
            self[0] = value
        else:
            raise TensorError(".v can only be used on scalars")


class ComputeOp(Operation):
    """Compute operation"""

    def __init__(self, name, shape, fcompute, dtype, loc, tensor=None):
        super().__init__(name, loc)
        self.fcompute = fcompute
        self.shape = shape
        self.dtype = dtype
        self.name = name
        if tensor is None:  # hcl.compute, which creates a new tensor
            self.tensor = AllocOp(name, shape, dtype, loc)
            self.kind = "compute"
        elif (
            isinstance(tensor, str) and tensor == "no_alloc"
        ):  # hcl.mutate, which doesn't create a new tensor
            self.tensor = None
            self.kind = "mutate"
        elif isinstance(
            tensor, AllocOp
        ):  # hcl.update, which updates an existing tensor
            self.tensor = tensor
            self.kind = "update"
        else:
            raise HCLValueError("tensor must be either None, 'no_alloc', or an AllocOp")
        self.body = []
        self.iter_vars = []
        self.reduce_vars = []
        self.level = len(scope)
        # For stages that do not produce a tensor
        # we use an auxiliary tensor to attach loop axis
        self.aux_tensor = AllocOp(name, shape, dtype, loc)
        self.input_tensors = []

        # update tensor's reference to fcompute
        if self.tensor is not None:
            self.tensor.fcompute = fcompute
        self.aux_tensor.fcompute = fcompute

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"{self.name} = compute({self.shape}, {self.dtype})"
        code_str += " (" + ", ".join([i.name for i in self.iter_vars]) + ")"
        code_str += " {\n"
        for op in self.body:
            code_str += f"{op}\n"
        code_str = print_indent(code_str, self.level)
        code_str += "}\n"
        return code_str


class IfOp(Operation):
    def __init__(self, cond, loc):
        super().__init__("if", loc)
        self.cond = immediate_to_constant(cond, loc)
        self.body = []
        self.else_body = []
        self.level = len(scope)
        self.else_branch_valid = False

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"if {self.cond} {{\n"
        for stmt in self.body:
            code_str += f"{stmt}\n"
        code_str = print_indent(code_str, self.level)
        code_str += "}"
        if self.else_branch_valid:
            code_str += " else {\n"
            for stmt in self.else_body:
                code_str += f"{stmt}\n"
            code_str = print_indent(code_str, self.level)
            code_str += "}"
        return code_str


class ElseOp(Operation):
    def __init__(self, loc):
        super().__init__("else", loc)
        self.body = []
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += "else {\n"
        for stmt in self.body:
            code_str += f"{stmt}\n"
        code_str = print_indent(code_str, self.level)
        code_str += "}"
        return code_str


class ElseIfOp(Operation):
    def __init__(self, cond, loc):
        super().__init__("elseif", loc)
        self.cond = cond
        self.body = []
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"else if ({self.cond}) {{\n"
        for stmt in self.body:
            code_str += f"{stmt}\n"
        code_str = print_indent(code_str, self.level)
        code_str += "}"
        return code_str


class IterVar(Expr):
    """Iteration variable."""

    def __init__(self, name, parent_loop, loc):
        super().__init__(name, loc)
        self.parent_loop = parent_loop
        self.level = len(scope)
        self.dtype = Index()
        self.reusable = True

    def __repr__(self):
        return self.name


class ReduceVar(IterVar):
    """Reduction variable."""

    def __init__(self, name, parent_loop, loc, bound=None):
        super().__init__(name, parent_loop, loc)
        self.bound = bound

    @property
    def lower_bound(self):
        return self.bound[0]

    @property
    def upper_bound(self):
        return self.bound[1]


class ReturnOp(Operation):
    def __init__(self, expr, loc):
        super().__init__("return", loc)
        self.expr = expr
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        if isinstance(self.expr, AllocOp):
            code_str += f"return {self.expr.name}"
        else:
            code_str += f"return {self.expr}"
        return code_str


class ForOp(Operation):
    def __init__(self, tag, name, low, high, step, loc):
        super().__init__("for", loc)
        self.tag = tag
        self.name = name
        self.low = low
        self.high = high
        self.step = step
        self.body = []
        self.iter_var = IterVar(name, self, loc)
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"for ({self.name} = {self.low}; {self.name} < {self.high}; {self.name} += {self.step}) {{\n"
        for stmt in self.body:
            code_str += f"{stmt}\n"
        code_str = print_indent(code_str, self.level)
        code_str += "}"
        return code_str


class WhileOp(Operation):
    def __init__(self, cond, loc):
        super().__init__("while", loc)
        self.cond = cond
        self.body = []
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"while ({self.cond}) {{\n"
        for stmt in self.body:
            code_str += f"{stmt}\n"
        code_str = print_indent(code_str, self.level)
        code_str += "}"
        return code_str


class FuncOp(Operation):
    def __init__(self, name, args, body, loc):
        super().__init__("func", loc)
        self.name = name
        self.args = args
        self.body = body
        self.return_tensors = []
        self.level = len(scope)
        self.body_ip = None
        self.python_callable = None
        self.prototype = False

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        if self.prototype:
            # This is a function declaration
            code_str += f"func {self.name}({[v.name for v in self.args]})"
            code_str += f" -> ({[v.name for v in self.return_tensors]})\n"
            return code_str
        code_str += f"func {self.name}({[v.name for v in self.args]}) {{\n"
        for stmt in self.body:
            code_str += f"{stmt}\n"
        code_str = print_indent(code_str, self.level + 1)
        code_str += f"return {[v.name for v in self.return_tensors]}\n"
        code_str = print_indent(code_str, self.level)
        code_str += "}\n"
        return code_str


class CallOp(Expr):
    def __init__(self, name, args, rets, loc):
        super().__init__(name, loc)
        self.name = name
        self.args = args
        self.rets = rets
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str += print_indent(code_str, self.level)
        if len(self.rets) > 0:
            code_str += f"{[v.name for v in self.rets]} = "
        code_str += f"call {self.name}({[v.name for v in self.args]})"
        return code_str


@register_type_rules(select_rule)
class SelectOp(Expr):
    def __init__(self, cond, true_value, false_value, loc):
        super().__init__("select", loc)
        self.cond = cond
        self.true_value = true_value
        self.false_value = false_value
        self.level = len(scope)
        self.dtype = self.tinf_engine.infer(self)

    def __repr__(self):
        return f"({self.cond} ? {self.true_value} : {self.false_value})"


class StructConstructOp(Expr):
    def __init__(self, args, dtype, loc):
        super().__init__("struct", loc)
        self.args = args
        self.dtype = dtype
        self.level = len(scope)

    def __repr__(self):
        return f"({[str(v) for v in self.args]})"


class StructGetOp(Expr):
    def __init__(self, struct, field, loc):
        super().__init__("struct_get", loc)
        self.struct = struct
        self.field = field
        self.level = len(scope)
        self.dtype = self.tinf_engine.infer(self)

    def __repr__(self):
        return f"{self.struct}.{self.field}"


class ReduceOp(Expr):
    def __init__(self, name, expr, reduce_op, axis, dtype, init, loc):
        super().__init__("reduce", loc)
        self.name = name
        self.expr = expr
        self.scalar = AllocOp(name, (1,), dtype, loc)
        self.body = []
        self.reduce_op = reduce_op
        self.axis = axis
        self.dtype = dtype
        self.init = init
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str += "reduce"
        code_str += " (" + ", ".join([i.name for i in self.axis]) + ")"
        code_str += " {\n"
        for op in self.body:
            code_str += f"{op}\n"
        code_str = print_indent(code_str, self.level)
        code_str += "}\n"
        return code_str


class PrintOp(Operation):
    def __init__(self, args, fmt, loc):
        super().__init__("print", loc)
        self.args = args
        self.fmt = fmt
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += "print("
        code_str += ", ".join([str(v) for v in self.args])
        code_str += ", "
        code_str += repr(self.fmt)
        code_str += ")"
        return code_str


class PrintTensorOp(Operation):
    def __init__(self, tensor, loc):
        super().__init__("print_tensor", loc)
        self.tensor = tensor
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"print_tensor({self.tensor.name})"
        return code_str


# Customization Operations
class OpHandle(Expr):
    def __init__(self, name, loc):
        super().__init__("op_handle", loc)
        self.name = name
        self.level = len(scope)
        self.is_customize_op = True

    def __eq__(self, other):
        assert isinstance(other, OpHandle)
        return self.name == other.name

    def __repr__(self):
        return self.name


class LoopHandle(Expr):
    def __init__(self, op_hdl, name, loc):
        super().__init__("loop_handle", loc)
        self.op_hdl = op_hdl
        self.name = name
        self.level = len(scope)
        self.is_customize_op = True

    def __eq__(self, other):
        assert isinstance(other, LoopHandle)
        return self.name == other.name

    def __repr__(self):
        return self.name


class PartitionOp(Operation):
    def __init__(self, tensor, kind, dim, factor, loc):
        super().__init__("partition", loc)
        self.tensor = tensor
        self.kind = kind
        self.dim = dim
        self.factor = factor
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"hcl.partition({self.tensor.name}, kind={self.kind}, dim={self.dim}, factor={self.factor})"
        return code_str


class ReplaceOp(Operation):
    def __init__(self, src_tensor, dst_tensor, loc):
        super().__init__("replace", loc)
        self.src_tensor = src_tensor
        self.dst_tensor = dst_tensor
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"hcl.replace({self.src_tensor.name}, {self.dst_tensor.name})"
        return code_str


class ReshapeOp(Operation):
    def __init__(self, tensor, shape, loc):
        super().__init__("reshape", loc)
        self.tensor = tensor
        self.shape = shape
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"hcl.reshape({self.tensor.name}, {self.shape})"
        return code_str


class ReformOp(Operation):
    def __init__(self, target, layout, loc):
        super().__init__("reform", loc)
        self.target = target
        self.layout = layout
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"hcl.reform({self.target.name}, {self.layout})"
        return code_str


class ReuseAtOp(Operation):
    def __init__(self, target, axis, loc):
        super().__init__("reuse_at", loc)
        self.target = target
        self.axis = axis
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"hcl.reuse_at({self.target.name}, {self.axis})"
        return code_str


class BufferAtOp(Operation):
    def __init__(self, target, axis, loc):
        super().__init__("buffer_at", loc)
        self.target = target
        self.axis = axis
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"hcl.buffer_at({self.target.name}, {self.axis})"
        return code_str


class InterKernelToOp(Operation):
    def __init__(self, tensor, stage, fifo_depth, loc):
        super().__init__("inter_kernel_to", loc)
        self.tensor = tensor
        self.stage = stage
        self.fifo_depth = fifo_depth
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += (
            f"hcl.inter_kernel_to({self.tensor.name}, {self.stage}, {self.fifo_depth})"
        )
        return code_str


class OutlineOp(Operation):
    def __init__(self, stage_hdls, loc):
        super().__init__("outline", loc)
        self.stage_hdls = stage_hdls
        self.unify = None
        self.axis = None
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"hcl.outline({', '.join([v.name for v in self.stage_hdls])}, axis={self.axis}, unify={self.unify})"
        return code_str


class ReorderOp(Operation):
    def __init__(self, args, loc):
        super().__init__("reorder", loc)
        self.args = args
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"hcl.reorder({', '.join([v.name for v in self.args])})"
        return code_str


class SplitOp(Operation):
    def __init__(self, stage_hdl, parent, factor, loc):
        super().__init__("split", loc)
        self.parent = parent
        self.factor = factor
        self.results = [
            LoopHandle(stage_hdl, parent.name + ".outer", loc),
            LoopHandle(stage_hdl, parent.name + ".inner", loc),
        ]
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"{self.results[0].name}, {self.results[1].name} = hcl.split({self.parent.name}, factor={self.factor})"
        return code_str


class TileOp(Operation):
    def __init__(self, stage_hdl, x_parent, y_parent, x_factor, y_factor, loc):
        super().__init__("tile", loc)
        self.x_parent = x_parent
        self.y_parent = y_parent
        self.x_factor = x_factor
        self.y_factor = y_factor
        self.results = [
            LoopHandle(stage_hdl, x_parent.name + ".outer", loc),
            LoopHandle(stage_hdl, x_parent.name + ".inner", loc),
            LoopHandle(stage_hdl, y_parent.name + ".outer", loc),
            LoopHandle(stage_hdl, y_parent.name + ".inner", loc),
        ]
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"{self.results[0].name}, {self.results[1].name}, {self.results[2].name}, {self.results[3].name} = hcl.tile({self.x_parent.name}, {self.y_parent.name}, x_factor={self.x_factor}, y_factor={self.y_factor})"
        return code_str


class PipelineOp(Operation):
    def __init__(self, target, ii, loc):
        super().__init__("pipeline", loc)
        self.target = target
        self.ii = ii
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"hcl.pipeline({self.target.name}, {self.ii})"
        return code_str


class UnrollOp(Operation):
    def __init__(self, target, factor, loc):
        super().__init__("unroll", loc)
        self.target = target
        self.factor = factor
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"hcl.unroll({self.target.name}, {self.factor})"
        return code_str


class ParallelOp(Operation):
    def __init__(self, target, loc):
        super().__init__("parallel", loc)
        self.target = target
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"hcl.parallel({self.target.name})"
        return code_str


class FuseOp(Operation):
    def __init__(self, arg_list, loc):
        super().__init__("fuse", loc)
        self.arg_list = arg_list
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"hcl.fuse({', '.join([str(v) for v in self.arg_list])})"
        return code_str


class ComputeAtOp(Operation):
    def __init__(self, stage, parent, axis, loc):
        super().__init__("compute_at", loc)
        self.stage = stage
        self.parent = parent
        self.axis = axis
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += (
            f"hcl.compute_at({self.stage.name}, {self.parent.name}, {self.axis})"
        )
        return code_str


class SystolicOp(Operation):
    def __init__(self, target, loc):
        super().__init__("systolic", loc)
        self.target = target
        self.level = len(scope)
        self.is_customize_op = True

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"hcl.systolic({self.target.name})"
        return code_str


class AST:
    """HeteroCL AST

    HeteroCL AST is a hierarchical representation of the input program.
    It has a very simple model: a program is a list of operations.
    Each operation can optionally have a body, which is again a list of operations.
    """

    def __init__(self, top_func):
        self.region = [top_func]
        self.top_func = top_func

    def __repr__(self):
        code_str = ""
        for op in self.region:
            code_str += str(op)
        return code_str


class TypeInference:
    """A type inference engine for HeteroCL programs."""

    # pylint: disable=too-many-return-statements
    def infer(self, expr):
        """Infer the type of an expression"""
        if isinstance(expr, LoadOp):
            return self.infer_load(expr)
        if isinstance(expr, BinaryOp):
            return self.infer_binary(expr)
        if isinstance(expr, SelectOp):
            return self.infer_select(expr)
        if isinstance(expr, ConstantOp):
            return self.infer_const(expr)
        if isinstance(expr, IterVar):
            return Index()
        if isinstance(
            expr, (CastOp, BitCastOp, GetBitOp, GetSliceOp, StructConstructOp)
        ):
            return expr.dtype
        if isinstance(expr, ReduceOp):
            # TODO: infer the type of the reduction
            return expr.dtype
        if isinstance(expr, (SetBitOp, SetSliceOp, BitReverseOp)):
            return self.infer(expr.expr)
        if isinstance(expr, StructGetOp):
            assert isinstance(expr.struct.dtype, Struct)
            struct_t = expr.struct.dtype
            key_list = list(struct_t.dtype_dict.keys())
            key = key_list[expr.field]
            return struct_t.dtype_dict[key]
        if isinstance(expr, CallOp):
            return self.infer(expr.rets[0])
        if isinstance(expr, Neg):
            return self.infer(expr.expr)
        # math ops
        if isinstance(
            expr,
            (
                MathExpOp,
                MathPowOp,
                MathLogOp,
                MathLog2Op,
                MathLog10Op,
                MathSqrtOp,
                MathSinOp,
                MathCosOp,
                MathTanOp,
                MathTanhOp,
            ),
        ):
            return self.infer_math(expr)

        raise APIError(
            f"Type inference method not defined for expression of type: {type(expr)} in TypeInference.infer"
        )

    def infer_binary(self, expr):
        lhs_type = self.infer(expr.lhs)
        rhs_type = self.infer(expr.rhs)
        if isinstance(lhs_type, tuple):
            lhs_type = lhs_type[-1]
        if isinstance(rhs_type, tuple):
            rhs_type = rhs_type[-1]
        # find the rule set based on the operation type
        type_rule = get_type_rules(type(expr))
        res_type = type_rule(lhs_type, rhs_type)
        # MLIR limitation: modulo only supports integer <= 128 bits
        if isinstance(expr, Mod) and isinstance(res_type, (Int, UInt)):
            if res_type.bits > 128:
                DTypeWarning("Modulo only supports integer <= 128 bits").warn()
                res_type = Int(128) if isinstance(res_type, Int) else UInt(128)
        return res_type

    def infer_math(self, expr):
        input_type = self.infer(expr.expr)
        if isinstance(input_type, tuple):
            input_type = input_type[-1]
        # find the rule set based on the operation type
        type_rule = get_type_rules(type(expr))
        res_type = type_rule(input_type)
        return res_type

    def infer_select(self, expr):
        true_type = self.infer(expr.true_value)
        false_type = self.infer(expr.false_value)
        type_rule = get_type_rules(type(expr))
        res_type = type_rule(true_type, false_type)
        return res_type

    def infer_load(self, expr):
        return expr.tensor.dtype

    def infer_const(self, expr):
        return expr.dtype
