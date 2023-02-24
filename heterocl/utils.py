# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

import gc
import inspect
import sys
import os
import numpy as np

import hcl_mlir
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.ir import IntegerType, F16Type, F32Type, F64Type
from hcl_mlir.exceptions import DTypeError

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
    if isinstance(dtype, Index):
        return hcl_mlir.IndexType.get()
    if isinstance(dtype, Int):
        return IntegerType.get_signless(dtype.bits)
    if isinstance(dtype, UInt):
        if signless:
            return IntegerType.get_signless(dtype.bits)
        return IntegerType.get_unsigned(dtype.bits)
    if isinstance(dtype, Fixed):
        return hcl_d.FixedType.get(dtype.bits, dtype.fracs)
    if isinstance(dtype, UFixed):
        return hcl_d.UFixedType.get(dtype.bits, dtype.fracs)
    if isinstance(dtype, Float):
        if dtype.bits == 16:
            return F16Type.get()
        if dtype.bits == 32:
            return F32Type.get()
        if dtype.bits == 64:
            return F64Type.get()
    if isinstance(dtype, Struct):
        types = [hcl_dtype_to_mlir(t, signless) for t in dtype.dtype_dict.values()]
        return hcl_d.StructType.get(types)
    raise DTypeError(
        f"unknown type in hcl_dtype_to_mlir: {dtype} of type {type(dtype)}"
    )


# pylint: disable=inconsistent-return-statements
def get_mlir_dtype_str(dtype):
    if hcl_mlir.is_integer_type(dtype):
        if hcl_mlir.is_signed_type(dtype):
            return "int" + str(dtype.width)
        return "uint" + str(dtype.width)
    if hcl_mlir.is_floating_point_type(dtype):
        if isinstance(dtype, F16Type):
            return "float16"
        if isinstance(dtype, F32Type):
            return "float32"
        if isinstance(dtype, F64Type):
            return "float64"
        # if isinstance(dtype, F80Type):
        #     return "float80"
        # if isinstance(dtype, F128Type):
        #     return "float128"
        raise TypeError(f"unknown type: {dtype}")
    if hcl_mlir.is_fixed_type(dtype):
        if isinstance(dtype, hcl_d.FixedType):
            return "fixed" + str(dtype.width) + "_" + str(dtype.frac)
        if isinstance(dtype, hcl_d.UFixedType):
            return "ufixed" + str(dtype.width) + "_" + str(dtype.frac)
        raise TypeError(f"unknown type: {dtype}")
    if hcl_mlir.is_index_type(dtype):
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
            f"get_extra_type_hints input dtype must be a HeteroCL type, got {type(dtype)}"
        )
    if isinstance(dtype, Int):
        return "s"
    if isinstance(dtype, UInt):
        return "u"
    return "_"


def remove_moved_attr(module):
    def _visit_region(region):
        if hasattr(region, "blocks"):
            for block in region.blocks:
                for op in block.operations:
                    _visit_op(op)

    def _visit_op(op):
        if "moved" in op.attributes:
            del op.attributes["moved"]
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
        if dtype.bits == 1:
            np_dtype = np.bool_
        elif dtype.bits <= 8:
            np_dtype = np.int8
        elif dtype.bits <= 16:
            np_dtype = np.int16
        elif dtype.bits <= 32:
            np_dtype = np.int32
        elif dtype.bits <= 64:
            np_dtype = np.int64
        elif dtype.bits <= 128:
            np_dtype = np.int128
        elif dtype.bits <= 256:
            np_dtype = np.int256
        else:
            raise DTypeError(
                f"Integer width ({dtype}) too large, not supported by numpy"
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
        val = val * (2**dtype.fracs)
        val = np.fix(val) % sb

        def cast_func(x):
            return x if x < sb_limit else x - sb

        val = np.vectorize(cast_func)(val)
        np_dtype = np.int64
    elif isinstance(dtype, UFixed):
        sb = 1 << dtype.bits
        val = val * (2**dtype.fracs)
        val = np.fix(val) % sb
        np_dtype = np.int64
    else:
        raise DTypeError(f"Unrecognized data type: {dtype}")

    array = np.array(val, dtype=np_dtype)
    return array


def get_min_value(dtype):
    """
    Get the minimum value of a data type
    """
    if isinstance(dtype, Int):
        return -(1 << (dtype.bits - 1))
    if isinstance(dtype, UInt):
        return 0
    if isinstance(dtype, Float):
        # arith dialect does not support -inf
        # return -np.inf
        return -1e10
    if isinstance(dtype, Fixed):
        return -(1 << (dtype.bits - 1))
    if isinstance(dtype, UFixed):
        return 0
    raise DTypeError(f"Unrecognized data type: {dtype}")


def get_max_value(dtype):
    """
    Get the maximum value of a data type
    """
    if isinstance(dtype, Int):
        return (1 << (dtype.bits - 1)) - 1
    if isinstance(dtype, UInt):
        return (1 << dtype.bits) - 1
    if isinstance(dtype, Float):
        # limitation of arith dialect
        # return np.inf
        return 1e10
    if isinstance(dtype, Fixed):
        return (1 << (dtype.bits - 1)) - 1
    if isinstance(dtype, UFixed):
        return (1 << dtype.bits) - 1
    raise DTypeError(f"Unrecognized data type: {dtype}")
