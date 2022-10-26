# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2022 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #
import os
import sys
import inspect
from ..utils import get_src_loc


class Location(object):
    """ Filename and linenumber
    """
    def __init__(self, filename, lineno):
        self.filename = filename
        self.lineno = lineno

    def __str__(self):
        return f"{self.filename}:{self.lineno}"

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

    def __repr__(self):
        return self.name

    def __add__(self, other):
        return BinaryOp("+", self, other, self.loc)

    def __sub__(self, other):
        return BinaryOp("-", self, other, self.loc)

    def __mul__(self, other):
        return BinaryOp("*", self, other, self.loc)
    
    def __div__(self, other):
        return BinaryOp("/", self, other, self.loc)

    def __gt__(self, other):
        return BinaryOp(">", self, other, self.loc)

class UnaryOp(Expr):
    """Base class for all unary operations.

    Parameters
    ----------
    name : str
        The name of the operation

    """
    def __init__(self, name, expr, loc):
        super().__init__(name, loc)
        self.expr = expr

    def __repr__(self):
        return f"({self.name} {self.expr})"

class BinaryOp(Expr):
    """Base class for all binary operations.

    Parameters
    ----------
    name : str
        The name of the operation

    """
    def __init__(self, name, lhs, rhs, loc):
        super().__init__(name, loc)
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"({self.lhs} {self.name} {self.rhs})"

class TernaryOp(Expr):
    """Base class for all ternary operations.

    Parameters
    ----------
    name : str
        The name of the operation

    """
    def __init__(self, name, cond, lhs, rhs, loc):
        super().__init__(name, loc)
        self.cond = cond
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"({self.cond} ? {self.lhs} : {self.rhs})"


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

class GetItemOp(Expr):
    """ GetItem operation.
    """
    def __init__(self, tensor, index, loc):
        super().__init__("getitem", loc)
        self.tensor = tensor
        self.index = index

    def __repr__(self):
        return f"{self.tensor.name}[{self.index}]"

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
        return GetItemOp(self, indices, self.loc)


class ComputeOp(Operation):
    """Compute operation
    """
    def __init__(self, name, shape, body, dtype, loc):
        super().__init__(name, loc)
        self.body = body
        self.tensor = AllocOp(name, shape, dtype, loc)

    def __repr__(self):
        return inspect.getsourcelines(self.body)[0][0].strip()

class IfOp(Operation):
    def __init__(self, cond, loc):
        super().__init__('if', loc)
        self.cond = cond
        self.body = list()

    def __repr__(self):
        code_str = "if ({}) {{\n".format(self.cond)
        for stmt in self.body:
            code_str += f"  {stmt}\n"
        code_str += "}"
        return code_str