from . import tensor
from tvm.api import _IterVar
from tvm import var as _var
from tvm import placeholder as _placeholder
from tvm import _api_internal as _tvm_api

def var(name = "var", dtype = "int32"):
  return _var(name = name, dtype = dtype)

def placeholder(shape, name = "placeholder", dtype = "int32"):
  p = _placeholder(shape, name = name, dtype = "int32")
  return tensor.Tensor(p, dtype = dtype)

def compute(shape, fcompute, name = "compute", inline = True):
  code = fcompute.__code__
  args = code.co_varnames
  nargs = code.co_argcount
  indices = [_IterVar((0, shape[n]), args[n], 0) for n in range(0, nargs)]

  if inline:
    body = fcompute(*indices)
  else:
    pass

  op = _tvm_api._ComputeOp(name, "", indices, [body])
  print body
