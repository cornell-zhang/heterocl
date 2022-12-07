import hcl_mlir
from hcl_mlir.ir import *
from hcl_mlir.exceptions import HCLValueError

from .types import Float, Int
from .context import UniqueName
from .operation import placeholder
from .tensor import Tensor
from .utils import get_dtype_str
from .ast import ast
from .schedule import get_src_loc

def print(vals, format_str=""):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    if isinstance(vals, ast.AllocOp):
        printOp = ast.PrintTensorOp(vals, loc)
    elif isinstance(vals, (int, float, ast.Expr)):
        value = ast.immediate_to_constant(vals, loc)
        printOp = ast.PrintOp([value], format_str, loc)
    elif isinstance(vals, (tuple, list)):
        # When vals is an tuple
        if isinstance(vals, tuple):
            vals = list(vals)
        # Check if all elements are int/float or expression
        for i, val in enumerate(vals):
            if isinstance(val, int) or isinstance(val, float):
                continue
            elif isinstance(val, ast.Expr):
                continue
            else:
                raise HCLValueError(
                    "Unsupported type of element in tuple: {} of type {}"
                    .format(val, type(val)))
        # when vals is empty
        if len(vals) == 0:
            value = ast.immediate_to_constant(0, loc)
            vals = [value]
        # when v in vals is int or float
        for i, v in enumerate(vals):
            vals[i] = ast.immediate_to_constant(v, loc)
        printOp = ast.PrintOp(vals, format_str, loc)
    else:
        raise HCLValueError(f"Unsupported type for print: {vals}")
    region = ast.scope.get()
    region.append(printOp)