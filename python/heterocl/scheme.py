"""Quantization scheme."""
from . import types
from .debug import APIError

class Scheme():
    """A quantization scheme.

    To create a scheme, use ``heterocl.create_scheme``. A scheme has two
    methods: one is to downsize tensors to integer type and the other is to
    quantize tensors to non-integer type. The scheme should only be created by
    the API. Users should not directly call the constructor.

    Parameters
    ----------
    inputs : list of Tensor
        A list of input tensors to the scheme

    func : callable
        The algorithm definition

    Attributes
    ----------
    inputs : list of Tensor
        A list of input tensors to the scheme

    func : callable
        The algorithm definition

    dtype_dict : dict(str, Type)
        A dictionary that maps between a name and its data type


    See Also
    --------
    heterocl.create_scheme
    """

    current = None

    def __init__(self, inputs, func):
        self.inputs = inputs
        self.func = func
        self.dtype_dict = {}

    def downsize(self, inputs, dtype):
        if not isinstance(dtype, (types.Int, types.UInt)):
            raise APIError("Downsize to non-integer type is not allowed")
        if not isinstance(inputs, list):
            inputs = [inputs]
        for i in inputs:
            try:
                self.set_dtype(i.name_with_prefix, dtype)
            except AttributeError:
                self.set_dtype(i.name, dtype)

    def quantize(self, inputs, dtype):
        if not isinstance(dtype, (types.Fixed, types.UFixed)):
            raise APIError("Quantize to integer type is not allowed")
        if not isinstance(inputs, list):
            inputs = [inputs]
        for i in inputs:
            try:
                self.set_dtype(i.name_with_prefix, dtype)
            except AttributeError:
                self.set_dtype(i.name, dtype)

    def set_dtype(self, name, dtype):
        self.dtype_dict[name] = dtype
