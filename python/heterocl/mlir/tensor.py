from ..types import dtype_to_str
import numpy as np
import hcl_mlir
from hcl_mlir.dialects import memref


class Tensor(object):
    """A wrapper class for hcl-mlir TensorOp
    """

    def __init__(self, shape, dtype, name=""):
        self.tensor = hcl_mlir.TensorOp(
            shape, memref.AllocOp, dtype, name=name)
        self.shape = shape
        self.dtype = dtype
        self.name = name

    def __getattr__(self, key):
        if key == "tensor":
            return self.tensor
        elif key == "dtype":
            return self.dtype  # hcl.Type
        else:
            return self.tensor.__getattribute__(key)

    def __getitem__(self, indices):
        return self.tensor.__getitem__(indices)

    def __setitem__(self, indices, expr):
        self.tensor.__setitem__(indices, expr)


class Array(object):
    """A wrapper class for numpy array
    Differences between array and tensor:
    tensor is only a placeholder while array holds actual values
    """

    def __init__(self, np_array, dtype):
        self.dtype = dtype
        # Data type check
        hcl_dtype_str = dtype_to_str(dtype)
        correct_dtype = np.dtype(hcl_dtype_str)
        if np_array.dtype != correct_dtype:
            np_array = np_array.astype(correct_dtype)
        self.np_array = np_array

    def asnumpy(self):
        return self.np_array

    def unwrap(self):
        # TODO(Niansong): suppor unwrap fixed-point tensor here
        return self.np_array
