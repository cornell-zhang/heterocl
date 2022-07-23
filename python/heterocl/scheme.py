from . import types
from .schedule import Stage, create_schedule


def create_scheme(inputs, func):
    """Create a quantization scheme.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    return Scheme(inputs, func)


def create_schedule_from_scheme(scheme, name=""):
    """Create a schedule from a scheme.
    """
    return create_schedule(scheme.inputs, func=None, name=name)


class Scheme(object):
    """A quantization scheme.
    """

    def __init__(self, inputs, func):
        self.inputs = inputs
        self.func = func
        # execute and attach stages to function
        ret = func(*inputs)
        for op, stage in Stage._mapping:
            func.__setattr__(op.name, op)

    def downsize(self, inputs, dtype):
        """Downsize a (list of) tensor to the specified integer type.
        """
        if not isinstance(dtype, (types.Int, types.UInt)):
            raise RuntimeError("Downsize to non-integer type is not allowed")
        if not isinstance(inputs, list):
            inputs = [inputs]
        for tensor in inputs:
            tensor.dtype = dtype

    def quantize(self, inputs, dtype):
        """Quantize a (list of) tensor to the specified fixed-point type.
        """
        if not isinstance(dtype, (types.Fixed, types.UFixed)):
            raise RuntimeError("Quantize to integer type is not allowed")
        if not isinstance(inputs, list):
            inputs = [inputs]
        for tensor in inputs:
            tensor.dtype = dtype
