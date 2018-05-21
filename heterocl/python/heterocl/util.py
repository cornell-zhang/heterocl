from tvm import make as _make
from tvm.expr import Var, Call
from tvm.api import _IterVar, decl_buffer
from tvm.tensor import Tensor, TensorSlice

def true():
  return _make.UIntImm("uint1", 1)

def make_for(indices, body, level):
    iter_var = indices[level]
    if level == len(indices) - 1:
      return _make.For(iter_var.var, iter_var.dom.min, iter_var.dom.extent, 0, 0, body)
    else:
      return _make.For(iter_var.var, iter_var.dom.min, iter_var.dom.extent, 0, 0, make_for(indices, body, level+1))

# return (index, bit, _)
def get_index(shape, args, level):
  if level == len(args) - 1: # the last arg
    if level == len(shape): # bit-selection
      return (0, args[level], 1)
    else:
      return (args[level], None, shape[level])
  else:
    index = get_index(shape, args, level+1)
    new_arg = args[level]
    new_index = _make.Add(index[0],
        _make.Mul(new_arg, index[2], False), False)
    new_acc = _make.Mul(index[2], shape[level], False)
    return (new_index, index[1], new_acc)

def get_type(dtype):
  if dtype[0:3] == "int":
    return "int", int(dtype[3:])
  elif dtype[0:4] == "uint":
    return "uint", int(dtype[4:])
  elif dtype[0:5] == "float":
    return "float", int(dtype[5:])
  else:
    raise ValueError("Unkown data type: " + dtype)

VID = 0
PID = 0
LID = 0
CID = 0
UID = 0
BID = 0
MID = 0
KID = 0
