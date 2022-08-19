import hcl_mlir
from hcl_mlir.ir import *

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
        # TODO(Niansong): setup default format string
        value = hcl_mlir.ConstantOp(get_dtype_str(dtype), vals)
        printOp = hcl_mlir.PrintOp([value])
    elif isinstance(vals, hcl_mlir.build_ir.ExprOp):
        # When vals is an expression
        printOp = hcl_mlir.PrintOp([vals])
    elif isinstance(vals, tuple):
        # When vals is an tuple
        printOp = hcl_mlir.PrintOp([*vals])
    # Attach format string as an attribute
    if format_str != "":
        printOp.built_op.attributes["format"] = StringAttr.get(format_str)
    return printOp
