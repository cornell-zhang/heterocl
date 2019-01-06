"""Quantization scheme."""
#pylint: disable=missing-docstring
from . import types
from .debug import APIError

class Scheme(object):
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

    Examples
    --------
    .. code-block:: python

        # example 1 - downsize
        hcl.init(hcl.Int(32))
        A = hcl.placeholder((10,))
        def kernel(A):
            return hcl.compute(A.shape, lambda x: A[x]+1, "B")
        s = hcl.create_scheme(A, kernel)
        # downsize tensor B to a 4-bit unsigned integer
        s.downsize(kernel.B, hcl.UInt(4))

        # example 1 - quantize
        hcl.init(hcl.Float())
        A = hcl.placeholder((10,))
        def kernel(A):
            return hcl.compute(A.shape, lambda x: A[x]+1, "B")
        s = hcl.create_scheme(A, kernel)
        # quantize tensor B to a 4-bit unsigned fixed point
        s.quantize(kernel.B, hcl.Fixed(4, 2))
    """

    current = None
    """The current scheme."""

    def __init__(self, inputs, func):
        self.inputs = inputs
        self.func = func
        self.dtype_dict = {}

    def downsize(self, inputs, dtype):
        """Downsize a (list of) tensor to the specified integer type.

        Parameters
        ----------
        inputs : Tensor of list of Tensor
            The tensor(s) to be downsized

        dtype : Type
            The target data type
        """
        if not isinstance(dtype, (types.Int, types.UInt)):
            raise APIError("Downsize to non-integer type is not allowed")
        if not isinstance(inputs, list):
            inputs = [inputs]
        for i in inputs:
            try:
                self._set_dtype(i.name_with_prefix, dtype)
            except AttributeError:
                self._set_dtype(i.name, dtype)

    def quantize(self, inputs, dtype):
        """Quantize a (list of) tensor to the specified fixed-point type.

        Parameters
        ----------
        inputs : Tensor of list of Tensor
            The tensor(s) to be quantized

        dtype : Type
            The target data type
        """
        if not isinstance(dtype, (types.Fixed, types.UFixed)):
            raise APIError("Quantize to integer type is not allowed")
        if not isinstance(inputs, list):
            inputs = [inputs]
        for i in inputs:
            try:
                self._set_dtype(i.name_with_prefix, dtype)
            except AttributeError:
                self._set_dtype(i.name, dtype)

    def _set_dtype(self, name, dtype):
        self.dtype_dict[name] = dtype
