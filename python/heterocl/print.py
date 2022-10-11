import hcl_mlir
from hcl_mlir.ir import *
from hcl_mlir.exceptions import HCLValueError

from .types import Float, Int
from .context import UniqueName
from .operation import placeholder
from .tensor import Tensor
from .utils import get_dtype_str

def print(vals, format_str=""):
    if isinstance(vals, Tensor):
        printOp = hcl_mlir.PrintMemRefOp(vals, get_dtype_str(vals.dtype))
    elif isinstance(vals, int) or isinstance(vals, float):
        dtype = Int(64) if isinstance(vals, int) else Float(64)
        value = hcl_mlir.ConstantOp(get_dtype_str(dtype), vals)
        if format_str == "":
            format_str = "%d \n" if isinstance(vals, int) else "%.3f \n"
        printOp = hcl_mlir.PrintOp([value], format_str)
    elif isinstance(vals, hcl_mlir.build_ir.ExprOp):
        # When vals is an expression
        if format_str == "":
            if "int" in get_dtype_str(vals.dtype):
                format_str = "%d \n"
            else:
                format_str = "%.3f \n"
        printOp = hcl_mlir.PrintOp([vals], format_str)
    elif isinstance(vals, (tuple, list)):
        # When vals is an tuple
        if isinstance(vals, tuple):
            vals = list(vals)
        # Check if all elements are int/float or expression
        for i, val in enumerate(vals):
            if isinstance(val, int) or isinstance(val, float):
                continue
            elif isinstance(val, hcl_mlir.build_ir.ExprOp):
                continue
            else:
                raise HCLValueError(
                    "Unsupported type of element in tuple: {} of type {}"
                    .format(val, type(val)))
        # when vals is empty
        if len(vals) == 0:
            value = hcl_mlir.ConstantOp('int32', 0)
            vals = [value]
        if format_str == "":
            for v in vals:
                if isinstance(v, int):
                    format_str += "%d "
                elif isinstance(v, float):
                    format_str += "%.3f "
                elif "int" in get_dtype_str(v.dtype):
                    format_str += "%d "
                else:
                    format_str += "%.3f "
            format_str += "\n"
        # when v in vals is int or float
        for i, v in enumerate(vals):
            if isinstance(v, int) or isinstance(v, float):
                dtype = Int(64) if isinstance(v, int) else Float(64)
                vals[i] = hcl_mlir.ConstantOp(get_dtype_str(dtype), v)
        printOp = hcl_mlir.PrintOp(vals, format_str)
    else:
        raise HCLValueError(f"Unsupported type for print: {vals}")
    return printOp
