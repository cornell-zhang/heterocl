import tvm
from . import tensor
from tvm import _api_internal as _tvm_api

def var(name = "var", dtype = "int32"):
  return tvm.var(name = name, dtype = dtype)

def placeholder(shape, name = "placeholder", dtype = "int32"):
  p = tvm.placeholder(shape, name = name, dtype = "int32")
  return tensor.Tensor(p, dtype = dtype)

def compute(shape, fcompute, name = "compute", inline = True):

