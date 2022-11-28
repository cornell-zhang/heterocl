from .ir import intermediate as itmd
from .utils import get_src_loc

def exp(x):
    filename, lineno = get_src_loc()
    loc = itmd.Location(filename, lineno)
    return itmd.MathExpOp(x, loc)

def power(x, y):
    filename, lineno = get_src_loc()
    loc = itmd.Location(filename, lineno)
    return itmd.MathPowOp(x, y, loc)

def log(x):
    filename, lineno = get_src_loc()
    loc = itmd.Location(filename, lineno)
    return itmd.MathLogOp(x, loc)

def log2(x):
    filename, lineno = get_src_loc()
    loc = itmd.Location(filename, lineno)
    return itmd.MathLog2Op(x, loc)

def log10(x):
    filename, lineno = get_src_loc()
    loc = itmd.Location(filename, lineno)
    return itmd.MathLog10Op(x, loc)

def sqrt(x):
    filename, lineno = get_src_loc()
    loc = itmd.Location(filename, lineno)
    return itmd.MathSqrtOp(x, loc)

def sin(x):
    filename, lineno = get_src_loc()
    loc = itmd.Location(filename, lineno)
    return itmd.MathSinOp(x, loc)

def cos(x):
    filename, lineno = get_src_loc()
    loc = itmd.Location(filename, lineno)
    return itmd.MathCosOp(x, loc)

def tanh(x):
    filename, lineno = get_src_loc()
    loc = itmd.Location(filename, lineno)
    return itmd.MathTanhOp(x, loc)