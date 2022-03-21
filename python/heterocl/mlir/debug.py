import hcl_mlir
from hcl_mlir.ir import *

from ..types import Float, Int
from .context import UniqueName
from .operation import placeholder
from .tensor import Tensor
from .utils import get_dtype_str


def print(vals, format_str=""):
    if isinstance(vals, Tensor):
        printOp = hcl_mlir.PrintOp(vals, get_dtype_str(vals.dtype))
    elif isinstance(vals, int) or isinstance(vals, float):
        # create a memref and store the number in it
        dtype = Int(64) if isinstance(vals, int) else Float(64)
        single_tensor = placeholder(
            (1,), name=UniqueName.get("scalar"), dtype=dtype)
        index = hcl_mlir.ConstantOp("index", 0)
        value = hcl_mlir.ConstantOp(get_dtype_str(dtype), vals)
        hcl_mlir.StoreOp(value, single_tensor.op, [index])
        printOp = hcl_mlir.PrintOp(single_tensor, get_dtype_str(dtype))
    elif isinstance(vals, hcl_mlir.build_ir.ExprOp):
        # When vals is an expression
        single_tensor = placeholder(
            (1,), name=UniqueName.get("scalar"), dtype=get_dtype_str(vals.dtype)
        )
        index = hcl_mlir.ConstantOp("index", 0)
        hcl_mlir.StoreOp(vals, single_tensor.op, [index])
        printOp = hcl_mlir.PrintOp(single_tensor, get_dtype_str(vals.dtype))
    elif isinstance(vals, tuple):
        # When vals is an tuple
        pass
    # Attach format string as an attribute
    if format_str != "":
        printOp.built_op.attributes["format"] = StringAttr.get(format_str)
    return printOp
