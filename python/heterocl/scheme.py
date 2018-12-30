from . import types
from . import debug

class Scheme():

    current = None

    def __init__(self, inputs, func):
        self.inputs = inputs
        self.func = func
        self.dtype_dict = {}

    def downsize(self, inputs, dtype):
        assert isinstance(dtype, (types.Int, types.UInt))
        if not isinstance(inputs, list):
            inputs = [inputs]
        for i in inputs:
            try:
                self.set_dtype(i.name_with_prefix, dtype)
            except AttributeError:
                self.set_dtype(i.name, dtype)

    def quantize(self, inputs, dtype):
        assert isinstance(dtype, (types.Fixed, types.UFixed))
        if not isinstance(inputs, list):
            inputs = [inputs]
        for i in inputs:
            try:
                self.set_dtype(i.name_with_prefix, dtype)
            except AttributeError:
                self.set_dtype(i.name, dtype)

    def set_dtype(self, name, dtype):
        self.dtype_dict[name] = dtype
