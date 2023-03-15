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


def make_anywidth_numpy_array(val, bitwidth):
    """
    Converts a numpy array to any target bitwidth.
    ----------------
    Parameters:
    val: numpy.ndarray
        numpy array, can be any numpy native bitwidth, e.g. np.int64
    bitwidth: int
        target bitwidth e.g. 9, 31, 198
    signed: True or False
        whether the values in the array are signed or unsigned
    ----------------
    Returns:
    numpy.ndarray
        numpy array with the target bitwidth
    """
    shape = val.shape
    sign_array = val > 0
    avail_bytes = val.itemsize  # number of bytes of each element
    # The following code has several steps to convert the numpy array to have
    # the correct data type in order to create an MLIR constant tensor.
    # Since MLIR-NumPy Python interface only supports byte-addressable data types,
    # we need to change the data type of the array to have the minimum number of bytes
    # that can represent the target bitwidth.
    # e.g., hcl.const_tensor(arr, dtype=hcl.Int(20)) (6*6 array)
    #       which requires 20 bits (3 bytes) to represent each element
    # declaration: 6*6*i20
    # numpy input: 6*6*i64
    # 1. Decompose the original i32 or i64 array into a structured array of uint8
    #  -> decompose: 6*6*8*i8
    # pylint: disable=no-else-return
    # I think this if-else makes the code more readable
    if bitwidth == 1:
        return np.packbits(val, axis=None, bitorder="little")
    else:
        # Here we construct a customized NumPy dtype, "f0", "f1", "f2", etc.
        # are the field names, and the entire data type is `op.values.dtype`.
        # This can be viewed as a `union` type in C/C++.
        # Please refer to the documentation for more details:
        # https://numpy.org/doc/stable/reference/arrays.dtypes.html#specifying-and-constructing-data-types
        decomposed_np_dtype = np.dtype(
            (
                val.dtype,
                {f"f{i}": (np.uint8, i) for i in range(val.dtype.itemsize)},
            )
        )
        val = val.view(decomposed_np_dtype)
        # 2. Compose the uint8 array into a structured array of target bitwidth
        # This is done by taking the first several bytes of the uint8 array
        # "u1" means one unsigned byte, and "i1" means one signed byte
        # f0 is LSB, fn is MSB
        n_bytes = int(np.ceil(bitwidth / 8))
        new_dtype = np.dtype(
            {
                "names": [f"f{i}" for i in range(n_bytes)],
                "formats": ["u1"] * n_bytes,
                "offsets": list(range(n_bytes)),
                "itemsize": n_bytes,
            }
        )
        # sometimes the available bytes are not enough to represent the target bitwidth
        # so that we need to pad the array
        _bytes = [val[f"f{i}"] for i in range(min(avail_bytes, n_bytes))]
        if avail_bytes < n_bytes:
            padding = np.where(sign_array, 0x00, 0xFF).astype(np.uint8)
            _bytes += [padding] * (n_bytes - avail_bytes)
        # -> compose: 6*6*3*i8
        val = np.stack(_bytes, axis=-1)
        # -> flatten: 108*i8
        val = val.flatten()
        # -> view: 36*i24
        val = val.view(np.dtype(new_dtype))
        # -> reshape: 6*6*i24
        val = val.reshape(shape)
        return val
