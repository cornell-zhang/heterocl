import hcl_mlir
from hcl_mlir.ir import *
from hcl_mlir.dialects import hcl as hcl_d
from ..types import Int, UInt, Fixed, UFixed, Float

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
