import hcl_mlir
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.ir import *

from ..config import init_dtype
from ..types import Fixed, Float, Int, Type, UFixed, UInt, Struct, dtype_to_str


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
    elif isinstance(dtype, Struct):
        types = [hcl_dtype_to_mlir(t) for t in dtype.dtype_dict.values()]
        return hcl_d.StructType.get(types)
    else:
        raise RuntimeError("Not supported type")


def get_mlir_dtype_str(dtype):
    if hcl_mlir.is_integer_type(dtype):
        if hcl_mlir.is_signed_type(dtype):
            return "int" + str(dtype.width)
        else:
            return "uint" + str(dtype.width)
    elif hcl_mlir.is_floating_point_type(dtype):
        if isinstance(dtype, hcl_mlir.ir.F16Type):
            return "float16"
        elif isinstance(dtype, hcl_mlir.ir.F32Type):
            return "float32"
        elif isinstance(dtype, hcl_mlir.ir.F64Type):
            return "float64"
        elif isinstance(dtype, hcl_mlir.ir.F80Type):
            return "float80"
        elif isinstance(dtype, hcl_mlir.ir.F128Type):
            return "float128"
        else:
            raise TypeError(f"unknown type: {dtype}")
    elif hcl_mlir.is_fixed_type(dtype):
        if isinstance(dtype, hcl_d.FixedType):
            return "fixed" + str(dtype.width) + "_" + str(dtype.frac)
        else:
            return "ufixed" + str(dtype.width) + "_" + str(dtype.frac)
    elif hcl_mlir.is_index_type(dtype):
        return "int32"


def get_dtype_str(dtype):
    if hcl_mlir.is_hcl_mlir_type(dtype):
        return get_mlir_dtype_str(dtype)
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
