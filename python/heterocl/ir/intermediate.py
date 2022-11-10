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

    def __sub__(self, other):
        return Sub(self, other, self.loc)

    def __mul__(self, other):
        return Mul(self, other, self.loc)
    
    def __div__(self, other):
        return Div(self, other, self.loc)

    def __gt__(self, other):
        return BinaryOp(">", self, other, self.loc)

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

class Mod(BinaryOp):
    """Modulo operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__("%", lhs, rhs, loc)

class Cmp(BinaryOp):
    """Comparison operation.
    """
    def __init__(self, op, lhs, rhs, loc):
        super().__init__(op, lhs, rhs, loc)

class LeftShiftOp(BinaryOp):
    """Left shift operation.
    """
    def __init__(self, lhs, rhs, loc):
        super().__init__("<<", lhs, rhs, loc)

class ConstantOp(Expr):
    """Constant operation.
    """
    def __init__(self, value, dtype, loc):
        super().__init__(str(value), loc)
        self.value = value
        self.dtype = dtype

class LoadOp(Expr):
    """ Load operation.
    """
    def __init__(self, tensor, index, loc):
        super().__init__("getitem", loc)
        self.tensor = tensor
        self.index = index
        self.dtype = tensor.dtype

    def __repr__(self):
        return f"{self.tensor.name}[{self.index}]"

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
        code_str += f"{self.tensor.name}[{self.index}] = {self.value}"
        return code_str

class GetBitOp(Expr):
    """Get bit operation
    """
    def __init__(self, tensor, index, loc):
        super().__init__("getbit", loc)
        self.tensor = tensor
        self.index = index
        self.dtype = tensor.dtype

    def __repr__(self):
        return f"{self.tensor.name}[{self.index}]"

class SetBitOp(Operation):
    """Set bit operation
    """
    def __init__(self, tensor, index, value, loc):
        super().__init__("setbit", loc)
        self.tensor = tensor
        self.index = index
        self.value = value
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"{self.tensor.name}[{self.index}] = {self.value}"
        return code_str

class GetSliceOp(Expr):
    """Get slice operation
    """
    def __init__(self, tensor, start, end, loc):
        super().__init__("getslice", loc)
        self.tensor = tensor
        self.start = start
        self.end = end
        self.dtype = tensor.dtype

    def __repr__(self):
        return f"{self.tensor.name}[{self.start}:{self.end}]"


class SetSliceOp(Operation):
    """Set slice operation
    """
    def __init__(self, tensor, start, end, value, loc):
        super().__init__("setslice", loc)
        self.tensor = tensor
        self.start = start
        self.end = end
        self.value = value
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += f"{self.tensor.name}[{self.start}:{self.end}] = {self.value}"
        return code_str


class TensorSlice(Expr):
    def __init__(self, full_shape, op, dtype, parent, indices, loc, name=None):
        super().__init__(op)
        self.op = op
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
                if not hasattr(index, "dtype"):
                    raise HCLValueError("{} doesn't have dtype".format(index))
                if not (isinstance(index.dtype, Int) or isinstance(index.dtype, UInt) or isinstance(index, IterVar)): 
                    raise HCLValueError("{} is not an integer type or index type".format(index))
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
                self.op,
                self.dtype,
                self.parent,
                self.indices + indices,
                self.name,
            )
        elif len(self.indices + indices) == len(self.full_shape):
            # format indices
            new_indices = []
            for index in self.indices + indices:
                if isinstance(index, int):
                    index = ConstantOp(IndexType.get(), index)
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
            for index in indices:
                if isinstance(index, int):
                    index = ConstantOp(IndexType.get(), index)
                new_indices.append(index)
            return StoreOp(self.parent, list(self.indices) + new_indices, expr, self.loc)
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
    
    def __repr__(self):
        return f"{self.name} = alloc({self.shape}, {self.dtype})"

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        # if we are slicing tensor
        if len(indices) < len(self.shape):
            return TensorSlice(
                self.shape, self.op, self.dtype, self, indices, self.loc, self.name
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


class ComputeOp(Operation):
    """Compute operation
    """
    def __init__(self, name, shape, body, dtype, loc):
        super().__init__(name, loc)
        self.body = body
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.tensor = AllocOp(name, shape, dtype, loc)
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += inspect.getsourcelines(self.body)[0][0].strip()
        return code_str


class IfOp(Operation):
    def __init__(self, cond, loc):
        super().__init__('if', loc)
        self.cond = cond
        self.body = list()
        self.level = len(scope)

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += "if ({}) {{\n".format(self.cond)
        for stmt in self.body:
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


class ForOp(Operation):
    def __init__(self, name, low, high, step, loc):
        super().__init__('for', loc)
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

class FuncOp(Operation):
    def __init__(self, name, args, body, loc):
        super().__init__('func', loc)
        self.name = name
        self.args = args
        self.body = body
        self.level = len(scope)
        self.body_ip = None

    def __repr__(self):
        code_str = ""
        code_str = print_indent(code_str, self.level)
        code_str += "func {}({}) {{\n".format(self.name, ", ".join([v.name for v in self.args]))
        for stmt in self.body:
            code_str += f"{stmt}\n"
        code_str = print_indent(code_str, self.level)
        code_str += "}"
        return code_str

class IR(object):
    def __init__(self):
        loc = Location("unknown", 0)
        self.top_func = FuncOp("top", [], [], loc)

    def add_op(self, op):
        self.top_func.body.append(op)

    def __repr__(self):
        return str(self.top_func)