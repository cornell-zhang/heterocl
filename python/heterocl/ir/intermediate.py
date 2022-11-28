# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2022 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #
import os
import sys
import inspect
from ..utils import get_src_loc, hcl_dtype_to_mlir
from hcl_mlir.exceptions import *
from ..context import *
from ..types import *

def print_indent(string, level):
    for _ in range(level):
        string += "  "
    return string

def immediate_to_constant(value, loc):
    if not isinstance(value, (int, float)):
        return value # pass through
    if isinstance(value, int):
        if value < 0xFFFFFFFF:
            return ConstantOp(value, Int(32), loc)
        else:
            return ConstantOp(value, Int(64), loc)
    else:
        return ConstantOp(value, Float(64), loc)

class Location(object):
    """ Filename and linenumber
    """
    def __init__(self, filename, lineno):
        self.filename = filename
        self.lineno = lineno

    def __str__(self):
        return f"{self.filename}:{self.lineno}"

class Scope(object):
    """ Scope class to manage operation insertion scopes."""
    def __init__(self):
        self.stack = list()

    def push(self, scope):
        """Push a new scope to the stack.

        Parameters
        ----------
        scope : a Python list
        """
        self.stack.append(scope)

    def pop(self):
        return self.stack.pop()

    def get(self):
        return self.stack[-1]

    def __repr__(self):
        return str(self.stack)

    def __len__(self):
        return len(self.stack)

scope = Scope()
# this list is for operations
# that are not enclosed in a top-level function
# in the case that there is a top-level function,
# this list will be poped and the operations will
# be inserted into the top-level function body scope
scope.push(list())

class Operation(object):
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

    def __repr__(self):
        return self.name

class Expr(object):
    """Base class for all expressions.

    Parameters
    ----------
    name : str
        The name of the expression

    """
    def __init__(self, name, loc):
        self.name = name
        self.loc = loc
        # When an expression is built, its result will be set
        self.result = None

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

    def __neg__(self):
        return Neg(self, self.loc)

    def __lshift__(self, other):
        return LeftShiftOp(self, other, self.loc)

    def __rshift__(self, other):
        return RightShiftOp(self, other, self.loc)

    def __and__(self, other):
        return And(self, other, self.loc)

    def __or__(self, other):
        return Or(self, other, self.loc)

    def __xor__(self, other):
        return XOr(self, other, self.loc)

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
        """Bit slicing and bit selection
        """
        if isinstance(indices, slice):
            lo, hi = indices.start, indices.stop
            if isinstance(lo, int) and isinstance(hi, int):
                if lo > hi:
                    raise APIError(
                        "Lower bound should be smaller than upper bound. Use `.reverse()` if you want to reverse the bits"
                    )
                elif lo == hi:
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
        # TODO(Niansong): not sure when this should be called
        # throw an error for now
        raise HCLNotImplementedError("equal is not implemented yet")

    def astype(self, dtype):
        return CastOp(self, dtype)

    def __getattr__(self, key):
        """Access a field of a struct value
        """
         # bypass the attribute lookup to avoid infinite recursion
        if key in self.__dict__.keys():
            return self.__dict__[key]
        elif isinstance(self, LoadOp):
            # access a field from a struct tensor
            key_list = [k for k in self.tensor.dtype.dtype_dict.keys()]
            if key not in key_list:
                raise HCLValueError("No such field: " + key)
            key_idx = key_list.index(key)
            return StructGetOp(self, key_idx, self.loc)
        else:
            # We don't throw an error here
            # because the user may be trying to test if
            # an attribute exists with hasattr().
            return

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
        return f"({self.name} {self.expr} : {self.dtype})"

class Add(BinaryOp):
    """Addition operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__("+", lhs, rhs, loc)

class Sub(BinaryOp):
    """Subtraction operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__("-", lhs, rhs, loc)

class Mul(BinaryOp):
    """Multiplication operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__("*", lhs, rhs, loc)

class Div(BinaryOp):
    """Division operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__("/", lhs, rhs, loc)

class FloorDiv(BinaryOp):
    """Floor division operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__("//", lhs, rhs, loc)

class Mod(BinaryOp):
    """Modulo operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__("%", lhs, rhs, loc)

class LeftShiftOp(BinaryOp):
    """Left shift operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__("<<", lhs, rhs, loc)

class RightShiftOp(BinaryOp):
    """Right shift operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__(">>", lhs, rhs, loc)

class Cmp(BinaryOp):
    """Comparison operation.
    """
    def __init__(self, op, lhs, rhs, loc):
        super().__init__(op, lhs, rhs, loc)

class And(BinaryOp):
    """Bitwise and operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__("&", lhs, rhs, loc)

class Or(BinaryOp):
    """Bitwise or operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__("|", lhs, rhs, loc)

class XOr(BinaryOp):
    """Bitwise xor operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__("^", lhs, rhs, loc)

class Invert(UnaryOp):
    """Bitwise invert operation, e.g. 0b1011 -> 0b0100.
    """
    def __init__(self, expr, loc):
        super().__init__("~", expr, loc)

class Neg(UnaryOp):
    """Negate operation, i.e. -x for any expression x.
    """
    def __init__(self, expr, loc):
        super().__init__("neg", expr, loc)

class BitReverseOp(UnaryOp):
    """Bit reverse operation.
    """
    def __init__(self, expr, loc):
        super().__init__("bit_reverse", expr, loc)

class BitCastOp(UnaryOp):
    """Bit cast operation.
    """
    def __init__(self, expr, dtype, loc):
        super().__init__("bit_cast", expr, loc)
        self.dtype = dtype

class MathExpOp(UnaryOp):
    """Mathematical exponential operation.
    """
    def __init__(self, expr, loc):
        super().__init__("exp", expr, loc)

class MathPowOp(BinaryOp):
    """Mathematical power operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__("pow", lhs, rhs, loc)

class MathLogOp(UnaryOp):
    """Mathematical log operation.
    """
    def __init__(self, expr, loc):
        super().__init__("log", expr, loc)

class MathLog2Op(UnaryOp):
    """Mathematical log2 operation.
    """
    def __init__(self, expr, loc):
        super().__init__("log2", expr, loc)

class MathLog10Op(UnaryOp):
    """Mathematical log10 operation.
    """
    def __init__(self, expr, loc):
        super().__init__("log10", expr, loc)

class MathSqrtOp(UnaryOp):
    """Mathematical square root operation.
    """
    def __init__(self, expr, loc):
        super().__init__("sqrt", expr, loc)

class MathSinOp(UnaryOp):
    """Mathematical sine operation.
    """
    def __init__(self, expr, loc):
        super().__init__("sin", expr, loc)

class MathCosOp(UnaryOp):
    """Mathematical cosine operation.
    """
    def __init__(self, expr, loc):
        super().__init__("cos", expr, loc)

class MathTanhOp(UnaryOp):
    """Mathematical tangent operation.
    """
    def __init__(self, expr, loc):
        super().__init__("tanh", expr, loc)

class LogicalAnd(BinaryOp):
    """Logical and operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__("&&", lhs, rhs, loc)

class LogicalOr(BinaryOp):
    """Logical or operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__("||", lhs, rhs, loc)

class LogicalXOr(BinaryOp):
    """Logical xor operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__("^^", lhs, rhs, loc)

class PrintOp(Operation):
    """Print operation.
    """
    def __init__(self, expr_list, loc, format_str=""):
        super().__init__("print", loc)
        self.expr_list = expr_list
        self.format_str = format_str

    def __repr__(self):
        code_str = ""
        code_str += print_indent(code_str, self.level)
        code_str += f"print({self.name} {self.expr_list}, fmt={self.format_str})"
        return code_str

class PrintMemRefOp(Operation):
    """Print memref operation.
    """
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
    """Constant scalar operation.
    """
    def __init__(self, value, dtype, loc):
        super().__init__(str(value), loc)
        self.value = value
        self.dtype = dtype

    def __repr__(self):
        return f"{self.value}"

class ConstantTensorOp(Expr):
    """Constant tensor operation.
    """
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
        code_str += f"{self.name} = constant_tensor({self.values.shape}, {self.values.dtype})"
        return code_str

class LoadOp(Expr):
    """ Load operation.
    """
    def __init__(self, tensor, index, loc):
        super().__init__("getitem", loc)
        self.tensor = tensor
        self.index = index
        self.dtype = tensor.dtype

    def __repr__(self):
        return f"{self.tensor.name}{self.index}"

class StoreOp(Operation):
    """ Store operation.
    """
    def __init__(self, tensor, index, value, loc):
        super().__init__("setitem", loc)
        self.tensor = tensor
        self.index = index
        self.value = value
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"{self.tensor.name}{self.index} = {self.value}"
        return code_str

class GetBitOp(Expr):
    """Get bit operation
    """
    def __init__(self, expr, index, loc):
        super().__init__("getbit", loc)
        self.expr = expr
        self.index = index

    def __repr__(self):
        return f"{self.expr}[{self.index}]"

class SetBitOp(Operation):
    """Set bit operation
    """
    def __init__(self, expr, index, value, loc):
        super().__init__("setbit", loc)
        self.expr = expr
        self.index = index
        self.value = value
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"{self.expr}[{self.index}] = {self.value}"
        return code_str

class GetSliceOp(Expr):
    """Get slice operation
    """
    def __init__(self, expr, start, end, loc):
        super().__init__("getslice", loc)
        self.expr = expr
        self.start = start
        self.end = end

    def __repr__(self):
        return f"{self.expr}[{self.start}:{self.end}]"


class SetSliceOp(Operation):
    """Set slice operation
    """
    def __init__(self, expr, start, end, value, loc):
        super().__init__("setslice", loc)
        self.expr = expr 
        self.start = start
        self.end = end
        self.value = value
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
        shape = list()
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
        if len(self.indices + indices) < len(self.full_shape):
            return TensorSlice(
                self.full_shape,
                self.dtype,
                self.parent,
                self.indices + indices,
                self.loc,
                self.name,
            )
        elif len(self.indices + indices) == len(self.full_shape):
            # format indices
            new_indices = []
            for index in self.indices + indices:
                index = immediate_to_constant(index, self.loc)
                new_indices.append(index)
            load = LoadOp(self.parent, new_indices, self.loc)
            return load
        else:
            raise TensorError("Indices length > # of array dimensions")

    def __setitem__(self, indices, expr):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(self.indices + indices) < len(self.full_shape):
            raise HCLNotImplementedError(
                "Writing to a slice of tensor is not allowed.")
        elif len(self.indices + indices) == len(self.full_shape):
            new_indices = []
            for index in list(self.indices) + list(indices):
                index = immediate_to_constant(index, self.loc)
                new_indices.append(index)
            expr = immediate_to_constant(expr, self.loc)
            store_op = StoreOp(self.parent, new_indices, expr, self.loc)
            region = scope.get()
            region.append(store_op)
        else:
            raise TensorError("Indices length > # of array dimensions," \
                + f"indices=[{self.indices + indices}], shape={self.full_shape}")

class AllocOp(Expr):
    """ Allocate memory for a buffer

    Parameters
    ----------
    name : str
        The name of the operation

    """
    def __init__(self, name, shape, dtype, loc):
        super().__init__(name, loc)
        self.shape = shape
        self.dtype = dtype
        # uses is a list of ComputeOp that uses the tensor produced by this op
        # we need such list to support create_schedule without an enclosing function
        self.uses = list()
        # Axes, a list of loop handles corresponding to the loop axes
        self.axis = list()
    
    def __repr__(self):
        return f"{self.name} = alloc({self.shape}, {self.dtype})"

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        # if we are slicing tensor
        if len(indices) < len(self.shape):
            return TensorSlice(
                full_shape=self.shape, dtype=self.dtype, 
                parent=self, indices=indices, loc=self.loc,
                name=self.name
            )
        # if we are loading a value from the tensor
        elif len(indices) == len(self.shape):
            # format indices
            new_indices = []
            for index in indices:
                index = immediate_to_constant(index, self.loc)
                new_indices.append(index)
            load = LoadOp(self, new_indices, self.loc)
            return load
        else:
            raise TensorError("Indices length > # of array dimensions")

    def __setitem__(self, indices, value):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) < len(self.shape):
            raise HCLNotImplementedError(
                "Writing to a slice of tensor is not allowed.")
        elif len(indices) == len(self.shape):
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
        else:
            raise TensorError(".v can only be used on scalars")
    
    @v.setter
    def v(self, value):
        if len(self.shape) == 1 and self.shape[0] == 1:
            value = immediate_to_constant(value, self.loc)
            self[0] = value
        else:
            raise TensorError(".v can only be used on scalars")


class ComputeOp(Operation):
    """Compute operation
    """
    def __init__(self, name, shape, fcompute, dtype, loc, tensor=None):
        super().__init__(name, loc)
        self.fcompute = fcompute
        self.shape = shape
        self.dtype = dtype
        self.name = name
        if tensor is None: # hcl.compute, which creates a new tensor
            self.tensor = AllocOp(name, shape, dtype, loc)
            self.kind = "compute"
        elif isinstance(tensor, str) and tensor == "no_alloc": # hcl.mutate, which doesn't create a new tensor
            self.tensor = None
            self.kind = "mutate"
        elif isinstance(tensor, AllocOp): # hcl.update, which updates an existing tensor
            self.tensor = tensor
            self.kind = "update"
        else:
            raise HCLValueError("tensor must be either None, 'no_alloc', or an AllocOp")
        self.body = list()
        self.iter_vars = list()
        self.reduce_vars = list()
        self.level = len(scope)
        # For stages that do not produce a tensor
        # we use an auxiliary tensor to attach loop axis
        self.aux_tensor = AllocOp(name, shape, dtype, loc)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"{self.name} = compute({self.shape}, {self.dtype}) {{\n"
        for op in self.body:
            code_str += f"{op}\n"
        code_str = print_indent(code_str, self.level)
        code_str += "}\n"
        return code_str


class IfOp(Operation):
    def __init__(self, cond, loc):
        super().__init__('if', loc)
        self.cond = cond
        self.body = list()
        self.else_body = list()
        self.level = len(scope)
        self.else_branch_valid = False

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += "if {} {{\n".format(self.cond)
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
        super().__init__('else', loc)
        self.body = list()
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
        super().__init__('elseif', loc)
        self.cond = cond
        self.body = list()
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += "else if ({}) {{\n".format(self.cond)
        for stmt in self.body:
            code_str += f"{stmt}\n"
        code_str = print_indent(code_str, self.level)
        code_str += "}"
        return code_str

class IterVar(Expr):
    """ Iteration variable.
    """
    def __init__(self, name, parent_loop, loc):
        super().__init__(name, loc)
        self.parent_loop = parent_loop
        self.level = len(scope)

    def __repr__(self):
        return self.name

class ReduceVar(IterVar):
    """ Reduction variable.
    """
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
        super().__init__('return', loc)
        self.expr = expr
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += "return {}".format(self.expr)
        return code_str

class ForOp(Operation):
    def __init__(self, tag, name, low, high, step, loc):
        super().__init__('for', loc)
        self.tag = tag
        self.name = name
        self.low = low
        self.high = high
        self.step = step
        self.body = list()
        self.iter_var = IterVar(name, self, loc)
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += "for ({} = {}; {} < {}; {} += {}) {{\n".format(
            self.name, self.low, self.name, self.high, self.name, self.step)
        for stmt in self.body:
            code_str += f"{stmt}\n"
        code_str = print_indent(code_str, self.level)
        code_str += "}"
        return code_str

class WhileOp(Operation):
    def __init__(self, cond, loc):
        super().__init__('while', loc)
        self.cond = cond
        self.body = list()
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += "while ({}) {{\n".format(self.cond)
        for stmt in self.body:
            code_str += f"{stmt}\n"
        code_str = print_indent(code_str, self.level)
        code_str += "}"
        return code_str

class FuncOp(Operation):
    def __init__(self, name, args, body, loc):
        super().__init__('func', loc)
        self.name = name
        self.args = args
        self.body = body
        self.return_tensors = list()
        self.level = len(scope)
        self.body_ip = None

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += "func {}({}) {{\n".format(self.name, ", ".join([v.name for v in self.args]))
        for stmt in self.body:
            code_str += f"{stmt}\n"
        code_str = print_indent(code_str, self.level + 1)
        code_str += "return {}\n".format(", ".join([v.name for v in self.return_tensors]))
        code_str = print_indent(code_str, self.level)
        code_str += "}"
        return code_str

class CallOp(Expr):
    def __init__(self, name, args, loc):
        super().__init__(name, loc)
        self.name = name
        self.args = args
        self.level = len(scope)

    def __repr__(self):
        return "{}({})".format(self.name, ", ".join([str(v) for v in self.args]))


class SelectOp(Expr):
    def __init__(self, cond, true_value, false_value, loc):
        super().__init__('select', loc)
        self.cond = cond
        self.true_value = true_value
        self.false_value = false_value
        self.level = len(scope)

    def __repr__(self):
        return "({} ? {} : {})".format(self.cond, self.true_value, self.false_value)


class StructConstructOp(Expr):
    def __init__(self, args, dtype, loc):
        super().__init__('struct', loc)
        self.args = args
        self.dtype = dtype
        self.level = len(scope)

    def __repr__(self):
        return "({})".format(", ".join([str(v) for v in self.args]))


class StructGetOp(Expr):
    def __init__(self, struct, field, loc):
        super().__init__('struct_get', loc)
        self.struct = struct
        self.field = field
        self.level = len(scope)

    def __repr__(self):
        return "{}.{}".format(self.struct, self.field)


class ReduceOp(Expr):
    def __init__(self, name, expr, reduce_op, axis, dtype, init, loc):
        super().__init__('reduce', loc)
        self.name = name
        self.expr = expr
        self.scalar = AllocOp(name, (1,), dtype, loc)
        self.reduce_op = reduce_op
        self.axis = axis
        self.dtype = dtype
        self.init = init
        self.level = len(scope)

    def __repr__(self):
        return "{}({}, {}, {}, {})".format(
            self.reduce_op, self.init, self.axis, self.dtype, self.name)


class SumOp(ReduceOp):
    def __init__(self, name, expr, axis, dtype, loc):
        super().__init__(name, expr, 'sum', axis, dtype, 0, loc)


class MinOp(ReduceOp):
    def __init__(self, name, expr, axis, dtype, loc):
        #TODO(Niansong): why init is 0x3F3F3F3F?
        super().__init__(name, expr, 'min', axis, dtype, 0x3F3F3F3F, loc)

class MaxOp(ReduceOp):
    def __init__(self, name, expr, axis, dtype, loc):
        super().__init__(name, expr, 'max', axis, dtype, 0x3F3F3F3F, loc)

class PrintOp(Operation):
    def __init__(self, args, fmt, loc):
        super().__init__('print', loc)
        self.args = args
        self.fmt = fmt
        self.level = len(scope)

    def __repr__(self):
        return "print({}, {})".format(", ".join([str(v) for v in self.args], self.fmt))

class PrintTensorOp(Operation):
    def __init__(self, tensor, loc):
        super().__init__('print_tensor', loc)
        self.tensor = tensor
        self.level = len(scope)

    def __repr__(self):
        return "print_tensor({})".format(self.tensor.name)

class IR(object):
    def __init__(self):
        loc = Location("unknown", 0)
        self.top_func = FuncOp("top", [], [], loc)

    def add_op(self, op):
        self.top_func.body.append(op)

    def __repr__(self):
        return str(self.top_func)