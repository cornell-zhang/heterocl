from tvm import make
from tvm.api import _IterVar
from tvm.tensor import Tensor as _Tensor
from tvm._ffi.node_generic import NodeGeneric

class Tensor(NodeGeneric):
  def __init__(self, t):
    self.tensor = None
    if (isinstance(t, _Tensor)):
      self.tensor = t

  def __repr__(self):
    return self.tensor.__repr__()

  def __getitem__(self, key):
    return self.tensor[key]

  def __setitem__(self, key, value):
    pass

  @property
  def body(self):
    return self.tensor

  @property
  def ndim(self):
    return self.tensor.ndim


def For(index, extent, func):

  min_val = extent[0]
  max_val = extent[1]
  diff = max_val - min_val

  code = func.__code__
  arg_names = code.co_varnames
  var = _IterVar((min_value, extent), arg_name, 0)
  body = func(var.var)
  print body
  node = make.For(var, min_value, extent, 0, 0, body)
  return node
