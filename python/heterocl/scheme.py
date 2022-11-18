from . import types
from .schedule import create_schedule_from_itmd, Pass
from .ir.intermediate import *


def create_scheme(inputs, func):
    """Create a quantization scheme.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    return Scheme(inputs, func)


def create_schedule_from_scheme(scheme, name=""):
    """Create a schedule from a scheme.
    """
    return create_schedule_from_itmd(scheme._itmd, scheme.inputs, scheme.func, name=name)

class AttachTensor(Pass):
    def __init__(self, scheme, itmd, func):
        super().__init__("attach_tensor", itmd)
        self.func = func
        self.scheme = scheme

    def apply(self):
        self.visit(self.itmd.top_func)

    def visit(self, op):
        self.attach_tensor(op, self.func)
        if hasattr(op, 'body'):
            for op in op.body:
                self.visit(op)

    def attach_tensor(self, op, func):
        if isinstance(op, ComputeOp):
            self.scheme._op_map[op.name] = op
            if op.kind == "compute":
                # attach op.tensor to func
                func.__setattr__(op.name, op.tensor)
            else:
                # attach op.aux_tensor to func
                func.__setattr__(op.name, op.aux_tensor)
        elif isinstance(op, AllocOp):
            if not hasattr(func, op.name):
                func.__setattr__(op.name, op)
                self.scheme._op_map[op.name] = op
        else:
            pass

class Scheme(object):
    """A quantization scheme.
    """

    def __init__(self, inputs, func):
        self.inputs = inputs
        self.func = func
        self._op_map = {} # op name -> op
        self._itmd = IR()
        self._itmd.top_func.args = inputs
        scope.pop()
        scope.push(self._itmd.top_func.body)
        ret = func(*inputs)
        if ret is None:
            outputs = list()
        elif isinstance(ret, tuple):
            outputs = list(ret)
        else:
            outputs = [ret]
        self._itmd.top_func.return_tensors.extend(outputs)
        attach_tensor_pass = AttachTensor(self, self._itmd, func)
        attach_tensor_pass.apply()

    def downsize(self, inputs, dtype):
        """Downsize a (list of) tensor to the specified integer type.
        """
        if not isinstance(dtype, (types.Int, types.UInt)):
            raise RuntimeError("Downsize to non-integer type is not allowed")
        if not isinstance(inputs, list):
            inputs = [inputs]
        for tensor in inputs:
            op = self._op_map[tensor.name]
            if isinstance(op, AllocOp):
                op.dtype = dtype
            elif isinstance(op, ComputeOp):
                op.tensor.dtype = dtype
                op.aux_tensor.dtype = dtype
                op.dtype = dtype
            else:
                raise HCLValueError(f"Unexpected op type: {type(op)} in Scheme._op_map, indexed by tensor: {tensor.name}")

    def quantize(self, inputs, dtype):
        """Quantize a (list of) tensor to the specified fixed-point type.
        """
        if not isinstance(dtype, (types.Fixed, types.UFixed)):
            raise RuntimeError("Quantize to integer type is not allowed")
        if not isinstance(inputs, list):
            inputs = [inputs]
        for tensor in inputs:
            op = self._op_map[tensor.name]
            if isinstance(op, AllocOp):
                op.dtype = dtype
            elif isinstance(op, ComputeOp):
                op.tensor.dtype = dtype
                op.aux_tensor.dtype = dtype
                op.dtype = dtype
            else:
                raise HCLValueError(f"Unexpected op type: {type(op)} in Scheme._op_map, indexed by tensor: {tensor.name}")

