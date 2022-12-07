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

def tanh(x):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    return ast.MathTanhOp(x, loc)