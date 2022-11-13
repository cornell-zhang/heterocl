import gc
import inspect
import sys
import os
import hcl_mlir
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.ir import *
from hcl_mlir.exceptions import *

from .config import init_dtype
from .types import Fixed, Float, Int, Type, UFixed, UInt, Struct, Index, dtype_to_str

def get_func_obj(func_name):
    for o in gc.get_objects():
        if inspect.isfunction(o) and o.__name__ == func_name:
            return o
    return None

def hcl_dtype_to_mlir(dtype, signless=False):
    if hcl_mlir.is_hcl_mlir_type(dtype):
        return dtype
    elif isinstance(dtype, Index):
        return hcl_mlir.IndexType.get()
    elif isinstance(dtype, Int):
        return IntegerType.get_signless(dtype.bits)
    elif isinstance(dtype, UInt):
        if signless:
            return IntegerType.get_signless(dtype.bits)
        else:
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
        types = [hcl_dtype_to_mlir(t, signless) for t in dtype.dtype_dict.values()]
        return hcl_d.StructType.get(types)
    else:
        raise DTypeError(f"unknown type in hcl_dtype_to_mlir: {dtype} of type {type(dtype)}")


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

def remove_moved_attr(module):
    def _visit_region(region):
        if hasattr(region, "blocks"):
            for block in region.blocks:
                for op in block.operations:
                    _visit_op(op)
    def _visit_op(op):
        if 'moved' in op.attributes:
            op.attributes.__delitem__('moved')
        if hasattr(op, 'body'):
            _visit_region(op.body)
    for func_op in module.body.operations:
        _visit_op(func_op)

def get_src_loc(frame=0):
    fr = sys._getframe(frame + 1) # +1 to ignore this function call
    return (os.path.basename(fr.f_code.co_filename), fr.f_lineno)