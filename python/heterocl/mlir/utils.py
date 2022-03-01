import hcl_mlir
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.ir import *

from ..config import init_dtype
from ..types import Fixed, Float, Int, Type, UFixed, UInt, dtype_to_str


def hcl_dtype_to_mlir(dtype):
    if hcl_mlir.is_hcl_mlir_type(dtype):
        return dtype
    elif isinstance(dtype, Int):
        return IntegerType.get_signless(dtype.bits)
    elif isinstance(dtype, UInt):
        return IntegerType.get_unsigned(dtype.bits)
    elif isinstance(dtype, Fixed):
        return hcl_d.FixedType.get(dtype.bits, dtype.fracs)
    elif isinstance(dtype, UFixed):
        return hcl_d.UFixedType.get(dtype.bits, dtype.fracs)
    elif isinstance(dtype, Float):
        if dtype.bits == 16:
            return F16Type.get()
        elif dtype.bits == 32:
            return F32Type.get()
        elif dtype.bits == 64:
            return F64Type.get()
    else:
        raise RuntimeError("Not supported type")


def get_dtype_str(dtype):
    if not dtype is None and not isinstance(dtype, (Type, str)):
        raise RuntimeError("Type error")
    dtype = init_dtype if dtype is None else dtype
    if not isinstance(dtype, str):
        dtype = dtype_to_str(dtype)
    return dtype


def get_extra_type_hints(dtype):
    """
    dtype: MLIR type
    """
    if not hcl_mlir.is_hcl_mlir_type(dtype):
        raise RuntimeError("Not MLIR type!")
    if isinstance(dtype, IntegerType):
        if dtype.is_unsigned:
            return "u"
        elif dtype.is_signed or dtype.is_signless:
            return "s"
    else:
        return "_"
