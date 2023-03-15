# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import math
from hcl_mlir.exceptions import DTypeError, APIError, DTypeWarning

from .types import dtype_to_str, Int, UInt, Float, Fixed, UFixed
from .utils import make_anywidth_numpy_array


class Array:
    """
    Represents a input tensor in HeteroCL.
    This class is a wrapper of numpy.ndarray, but it also
    support a wider range of data types, including any-width
    integer and fixed-point data types.
    """

    def __init__(self, array, dtype):
        """
        Parameters
        ----------
        array : numpy.ndarray or a python list
            The array to be wrapped.
            If the bitwidth of the data type is wider than 64,
            the array should be a python list.
        dtype : HeteroCL data type
        """
        self.dtype = dtype
        if dtype is None:
            raise APIError("Should provide type info")
        # self.np_array: a numpy array that holds the data
        # For float type, self.np_array is a float type numpy array
        # For int, uint, fixed, ufixed, self.np_array is a struct type numpy array
        # with each field being a byte.
        self.np_array = self._handle_overflow(array, dtype)
        if isinstance(dtype, (Int, UInt)):
            signed = isinstance(dtype, Int)
            # closest power of 2
            bitwidth = 1 << (self.dtype.bits - 1).bit_length()
            if bitwidth < 8: bitwidth = 8
            # this is to be compliant with MLIR's anywidth type representation
            # e.g. i1-i8 -> int8
            #      i9-i16 -> int16
            #      i17-i32 -> int32
            #      i33-i64 -> int64
            #      i65-i128 -> int128
            #      i129-i256 -> int256
            self.np_array = make_anywidth_numpy_array(self.np_array, bitwidth, signed)
    
    def asnumpy(self):
        """
        Convert HeteroCL array to numpy array / python list.
        If the bitwidth is wider than 64, the result will be a python list.
        Otherwise, return a numpy array.
        """
        if isinstance(self.dtype, Float):
            hcl_dtype_str = dtype_to_str(self.dtype)
            np_dtype = np.dtype(hcl_dtype_str)
            res_array = self.np_array.astype(np_dtype)
            return res_array
        elif isinstance(self.dtype, Int):
            if self.dtype.bits > 64:
                DTypeWarning(f"The bitwidth of target type is wider than 64 ({self.dtype}), .asnumpy() returns a python list")
            return self._struct_np_array_to_int()
        elif isinstance(self.dtype, UInt):
            if self.dtype.bits > 64:
                DTypeWarning(f"The bitwidth of target type is wider than 64 ({self.dtype}), .asnumpy() returns a python list")
            return self._struct_np_array_to_int()
        #TODO(Niansong): fixed/ufixed does not go through struct_np_array_to_int for now
        # because a change in IR is needed to support this, leaving it to another PR
        elif isinstance(self.dtype, Fixed):
            if self.dtype.bits > 64:
                DTypeWarning(f"The bitwidth of target type is wider than 64 ({self.dtype}), .asnumpy() returns a python list")
            # base_array = self._struct_np_array_to_int()
            # return base_array.astype(np.float64) / float(2 ** (self.dtype.fracs))
            return self.np_array.astype(np.float64) / float(2 ** (self.dtype.fracs))
        elif isinstance(self.dtype, UFixed):
            if self.dtype.bits > 64:
                DTypeWarning(f"The bitwidth of target type is wider than 64 ({self.dtype}), .asnumpy() returns a python list")
            # base_array = self._struct_np_array_to_int()
            # return base_array.astype(np.float64) / float(2 ** (self.dtype.fracs))
            return self.np_array.astype(np.float64) / float(2 ** (self.dtype.fracs))
        else:
            raise DTypeError(f"Unsupported data type {self.dtype}")

    def unwrap(self):
        return self.np_array


    def _handle_overflow(self, array, dtype):
        """
        If the dtype is wider than 64 bits,
        array should a list of numpy numbers.
        """
        # Data type check
        if isinstance(dtype, Float):
            if isinstance(array, list):
                array = np.array(array)
            hcl_dtype_str = dtype_to_str(dtype)
            correct_dtype = np.dtype(hcl_dtype_str)
            if array.dtype != correct_dtype:
                array = array.astype(correct_dtype)
        elif isinstance(dtype, Int):
            sb = 1 << self.dtype.bits
            sb_limit = 1 << (self.dtype.bits - 1)
            def cast_func(x):
                # recursive
                if isinstance(x, list):
                    return [cast_func(y) for y in x]
                # signed integer overflow function: wrap mode
                x = x % sb # cap the value to the max value of the bitwidth
                return x if x < sb_limit else x - sb
            if isinstance(array, list):
                array = [cast_func(x) for x in array] # TODO: this should be tested independently
            else:
                array = np.vectorize(cast_func)(array)
        elif isinstance(dtype, UInt):
            # Handle overflow
            sb = 1 << self.dtype.bits
            array = array % sb
        elif isinstance(dtype, Fixed):
            # Handle overflow
            sb = 1 << self.dtype.bits
            sb_limit = 1 << (self.dtype.bits - 1)
            array = array.astype(np.float64)
            array = array * (2**dtype.fracs)
            def cast_func(x):
                # recursive
                if isinstance(x, list):
                    return [cast_func(y) for y in x]
                x = math.trunc(x) % sb # rounds towards zero
                # signed integer overflow function: wrap mode
                return x if x < sb_limit else x - sb
            if isinstance(array, list):
                array = [cast_func(x) for x in array]
            else:
                array = np.vectorize(cast_func)(array)
            array = array.astype(np.int64)
        elif isinstance(dtype, UFixed):
            # Handle overflow
            sb = 1 << self.dtype.bits
            array = array.astype(np.float64)
            array = array * (2**dtype.fracs)
            def cast_func(x):
                # recursive
                if isinstance(x, list):
                    return [cast_func(y) for y in x]
                x = math.trunc(x) % sb # rounds towards zero
                return x
            if isinstance(array, list):
                array = [cast_func(x) for x in array]
            else:
                array = np.vectorize(cast_func)(array)
            array = array.astype(np.int64)
        else:
            raise DTypeError("Type error: unrecognized type: " + str(self.dtype))
        return array
    

    def _struct_np_array_to_int(self):
        pylist = self.np_array.tolist()
        # each element is a tuple
        def to_int(x):
            if isinstance(x, list):
                return [to_int(y) for y in x]
            signed = isinstance(self.dtype, (Int, Fixed))
            # turn x from tuple to list
            x = list(x)
            # find MSB
            byte_idx = (self.dtype.bits - 1) // 8
            bit_idx = (self.dtype.bits - 1) % 8
            msb = (x[byte_idx] & (1 << bit_idx)) > 0
            # sign extension
            if signed and msb:
                x[byte_idx] |= ((0xff << bit_idx) & 0xff)
                for i in range(byte_idx + 1, len(x)):
                    x[i] = 0xff
            # concatenate the tuple
            # each element is a byte
            byte_str = b''
            for i in range(len(x)):
                byte_str += x[i].to_bytes(1, byteorder='little', signed=False)
            value = int.from_bytes(byte_str, byteorder='little', signed=signed)
            return value
        pylist = to_int(pylist)
        if self.dtype.bits <= 64:
            return np.array(pylist, dtype=np.int64)
        else:
            return pylist

    def __repr__(self) -> str:
        return self.asnumpy().__repr__()
