from ..types import dtype_to_str
import numpy as np

class Tensor(object):
    """A wrapper class for numpy array
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