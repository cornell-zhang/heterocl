# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import MLIRLimitationError

from .ast import ast
from .utils import get_src_loc


def exp(x):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    return ast.MathExpOp(x, loc)


def power(x, y):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    return ast.MathPowOp(x, y, loc)


def log(x):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    return ast.MathLogOp(x, loc)


def log2(x):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    return ast.MathLog2Op(x, loc)


def log10(x):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    return ast.MathLog10Op(x, loc)


def sqrt(x):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    return ast.MathSqrtOp(x, loc)


def sin(x):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    return ast.MathSinOp(x, loc)


def cos(x):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    return ast.MathCosOp(x, loc)


def tan(x):
    raise MLIRLimitationError(
        "LLVM 15.0 does not support math.tan lowering."
        + " Please write tan as sin/cos for now. tan will be added in future releases."
    )
    # filename, lineno = get_src_loc()
    # loc = ast.Location(filename, lineno)
    # return ast.MathTanOp(x, loc)


def tanh(x):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    return ast.MathTanhOp(x, loc)
