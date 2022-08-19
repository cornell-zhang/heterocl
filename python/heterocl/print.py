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
        printOp = hcl_mlir.PrintOp([value])
        if format_str == "":
            format_str = "%d \n" if isinstance(vals, int) else "%.3f \n"
    elif isinstance(vals, hcl_mlir.build_ir.ExprOp):
        # When vals is an expression
        printOp = hcl_mlir.PrintOp([vals])
        if format_str == "":
            if "int" in get_dtype_str(vals.dtype):
                format_str = "%d \n"
            else:
                format_str = "%.3f \n"
    elif isinstance(vals, (tuple, list)):
        # When vals is an tuple
        if isinstance(vals, tuple):
            vals = list(vals)
        printOp = hcl_mlir.PrintOp(vals)
        if format_str == "":
            for v in vals:
                if "int" in get_dtype_str(v.dtype):
                    format_str += "%d "
                else:
                    format_str += "%.3f "
            format_str += "\n"
    else:
        raise HCLValueError(f"Unsupported type for print: {vals}")
    # Attach format string as an attribute
    if format_str != "":
        printOp.built_op.attributes["format"] = StringAttr.get(format_str)
    return printOp
