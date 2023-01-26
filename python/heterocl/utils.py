# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2023 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #

import gc
import inspect
import sys
import numpy as np
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
        raise DTypeError(
            f"unknown type in hcl_dtype_to_mlir: {dtype} of type {type(dtype)}"
        )


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
    dtype: HeteroCL type
    """
    if not isinstance(dtype, Type):
        raise TypeError(
            "get_extra_type_hints input dtype must be a HeteroCL type, got {}".format(
                type(dtype)
            )
        )
    if isinstance(dtype, Int):
        return "s"
    elif isinstance(dtype, UInt):
        return "u"
    else:
        return "_"


def remove_moved_attr(module):
    def _visit_region(region):
        if hasattr(region, "blocks"):
            for block in region.blocks:
                for op in block.operations:
                    _visit_op(op)

    def _visit_op(op):
        if "moved" in op.attributes:
            op.attributes.__delitem__("moved")
        if hasattr(op, "body"):
            _visit_region(op.body)

    for func_op in module.body.operations:
        _visit_op(func_op)


def get_src_loc(frame=0):
    fr = sys._getframe(frame + 1)  # +1 to ignore this function call
    return (os.path.basename(fr.f_code.co_filename), fr.f_lineno)


def make_const_tensor(val, dtype):
    # val is numpy ndarray
    if isinstance(dtype, (Int, UInt)):
        if dtype.bits <= 64:
            np_dtype = np.int64
        else:
            raise DTypeError(
                "Integer width ({}) too large, not supported by numpy".format(dtype)
            )
    elif isinstance(dtype, Float):
        if dtype.bits == 16:
            np_dtype = np.float16
        elif dtype.bits == 32:
            np_dtype = np.float32
        elif dtype.bits == 64:
            np_dtype = np.float64
        else:
            raise DTypeError("Unrecognized data type")
    elif isinstance(dtype, Fixed):
        sb = 1 << dtype.bits
        sb_limit = 1 << (dtype.bits - 1)
        val = val * (2 ** dtype.fracs)
        val = np.fix(val) % sb

        def cast_func(x):
            return x if x < sb_limit else x - sb

        val = np.vectorize(cast_func)(val)
        np_dtype = np.int64
    elif isinstance(dtype, UFixed):
        sb = 1 << dtype.bits
        val = val * (2 ** dtype.fracs)
        val = np.fix(val) % sb
        np_dtype = np.int64
    else:
        raise DTypeError("Unrecognized data type: {}".format(dtype))

    array = np.array(val, dtype=np_dtype)
    return array


def get_min_value(dtype):
    """
    Get the minimum value of a data type
    """
    if isinstance(dtype, Int):
        return -(1 << (dtype.bits - 1))
    elif isinstance(dtype, UInt):
        return 0
    elif isinstance(dtype, Float):
        # arith dialect does not support -inf
        # return -np.inf
        return -1e10
    elif isinstance(dtype, Fixed):
        return -(1 << (dtype.bits - 1))
    elif isinstance(dtype, UFixed):
        return 0
    else:
        raise DTypeError("Unrecognized data type: {}".format(dtype))

def get_max_value(dtype):
    """
    Get the maximum value of a data type
    """
    if isinstance(dtype, Int):
        return (1 << (dtype.bits - 1)) - 1
    elif isinstance(dtype, UInt):
        return (1 << dtype.bits) - 1
    elif isinstance(dtype, Float):
        # limitation of arith dialect
        # return np.inf
        return 1e10
    elif isinstance(dtype, Fixed):
        return (1 << (dtype.bits - 1)) - 1
    elif isinstance(dtype, UFixed):
        return (1 << dtype.bits) - 1
    else:
        raise DTypeError("Unrecognized data type: {}".format(dtype))