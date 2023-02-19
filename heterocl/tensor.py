# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from hcl_mlir.exceptions import DTypeError

from .types import dtype_to_str, Int, UInt, Float, Fixed, UFixed


class Array:
    """A wrapper class for numpy array
    Differences between array and tensor:
    tensor is only a placeholder while array holds actual values
    """

    def __init__(self, np_array, dtype):
        self.dtype = dtype  # should specify the type of `dtype`
        if isinstance(np_array, list):
            np_array = np.array(np_array)
        if dtype is not None:
            # Data type check
            if isinstance(dtype, Float):
                hcl_dtype_str = dtype_to_str(dtype)
                correct_dtype = np.dtype(hcl_dtype_str)
                if np_array.dtype != correct_dtype:
                    np_array = np_array.astype(correct_dtype)
            elif isinstance(dtype, Int):
                # Handle overflow
                sb = 1 << self.dtype.bits
                sb_limit = 1 << (self.dtype.bits - 1)
                np_array = np_array % sb

                def cast_func(x):
                    return x if x < sb_limit else x - sb

                vec_np_array = np.vectorize(cast_func)(np_array)
                np_array = vec_np_array.astype(np.uint64)
            elif isinstance(dtype, UInt):
                # Handle overflow
                sb = 1 << self.dtype.bits
                np_array = np_array % sb
                np_array = np_array.astype(np.uint64)
            elif isinstance(dtype, Fixed):
                # Handle overflow
                sb = 1 << self.dtype.bits
                sb_limit = 1 << (self.dtype.bits - 1)
                np_array = np_array * (2**dtype.fracs)
                np_array = np.fix(np_array) % sb

                def cast_func(x):
                    return x if x < sb_limit else x - sb

                vec_np_array = np.vectorize(cast_func)(np_array)
                np_array = vec_np_array.astype(np.uint64)
            elif isinstance(dtype, UFixed):
                # Handle overflow
                sb = 1 << self.dtype.bits
                np_array = np_array * (2**dtype.fracs)
                np_array = np.fix(np_array) % sb
                np_array = np_array.astype(np.uint64)
            else:
                raise DTypeError("Type error: unrecognized type: " + str(self.dtype))
        else:
            raise RuntimeError("Should provide type info")
        self.np_array = np_array

    def asnumpy(self):
        if isinstance(self.dtype, (Fixed, UFixed)):
            if isinstance(self.dtype, Fixed):
                res_array = self.np_array.astype(np.int64)
            else:
                res_array = self.np_array
            res_array = res_array.astype(np.float64) / float(2 ** (self.dtype.fracs))
            return res_array
        if isinstance(self.dtype, Int):
            res_array = self.np_array.astype(np.int64)
            return res_array
        if isinstance(self.dtype, Float):
            res_array = self.np_array.astype(float)
            return res_array
        return self.np_array

    def unwrap(self):
        return self.np_array

    def __repr__(self) -> str:
        return self.asnumpy().__repr__()
