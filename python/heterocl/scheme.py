from . import types
from .schedule import create_schedule_from_ast, Pass
from hcl_mlir.exceptions import *
from .ast import ast


def create_scheme(inputs, func):
    """Create a quantization scheme.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    return Scheme(inputs, func)


def create_schedule_from_scheme(scheme, name=""):
    """Create a schedule from a scheme.
    """
    return create_schedule_from_ast(scheme._ast, scheme.inputs, scheme.func, name=name)

class AttachTensor(Pass):
    def __init__(self, scheme, ast, func):
        super().__init__("attach_tensor", ast)
        self.func = func
        self.scheme = scheme

    def apply(self):
        self.visit(self.ast.top_func)

    def visit(self, op):
        self.attach_tensor(op, self.func)
        if hasattr(op, 'body'):
            for op in op.body:
                self.visit(op)

    def attach_tensor(self, op, func):
        if isinstance(op, ast.ComputeOp):
            self.scheme._op_map[op.name] = op
            if op.kind == "compute":
                # attach op.tensor to func
                func.__setattr__(op.name, op.tensor)
            else:
                # attach op.aux_tensor to func
                func.__setattr__(op.name, op.aux_tensor)
        elif isinstance(op, ast.AllocOp):
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
        self._ast = ast.IR()
        self._ast.top_func.args = inputs
        ast.scope.pop()
        ast.scope.push(self._ast.top_func.body)
        ret = func(*inputs)
        if ret is None:
            outputs = list()
        elif isinstance(ret, tuple):
            outputs = list(ret)
        else:
            outputs = [ret]
        self._ast.top_func.return_tensors.extend(outputs)
        attach_tensor_pass = AttachTensor(self, self._ast, func)
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
            if isinstance(op, ast.AllocOp):
                op.dtype = dtype
            elif isinstance(op, ast.ComputeOp):
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
            if isinstance(op, ast.AllocOp):
                op.dtype = dtype
            elif isinstance(op, ast.ComputeOp):
                op.tensor.dtype = dtype
                op.aux_tensor.dtype = dtype
                op.dtype = dtype
            else:
                raise HCLValueError(f"Unexpected op type: {type(op)} in Scheme._op_map, indexed by tensor: {tensor.name}")

