from . import types
from . import config
from tvm import make as _make
from tvm.expr import Var, Call
from tvm.api import _IterVar, decl_buffer
from tvm.tensor import Tensor, TensorSlice
import numbers
import sys

VID = 0
PID = 0
LID = 0
CID = 0
UID = 0
BID = 0
MID = 0
KID = 0
NID = 0

def true():
  return _make.UIntImm("uint1", 1)

def make_for(indices, body, level):
    iter_var = indices[level]
    if level == len(indices) - 1:
      body = _make.AttrStmt(iter_var, "loop_scope", iter_var.var, body)
      return _make.For(iter_var.var, iter_var.dom.min, iter_var.dom.extent, 0, 0, body)
    else:
      body = _make.AttrStmt(iter_var, "loop_scope", iter_var.var, make_for(indices, body, level+1))
      return _make.For(iter_var.var, iter_var.dom.min, iter_var.dom.extent, 0, 0, body)

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

def convert_dtype(dtype):
  dtype = config.init_dtype if dtype is None else dtype
  if isinstance(dtype, types.Type):
    if isinstance(dtype, types.Int):
      bits = dtype.bits
      if bits is None:
        return "int32"
      elif isinstance(bits, numbers.Number):
        return "int" + str(bits)
      elif isinstance(bits, (tuple, list)):
        return "int" + str(max(bits))
      else:
        raise ValueError("Unkown integer")
    elif isinstance(dtype, types.UInt):
      bits = dtype.bits
      if bits is None:
        return "uint32"
      elif isinstance(bits, numbers.Number):
        return "uint" + str(bits)
      elif isinstance(bits, (tuple, list)):
        return "uint" + str(max(bits))
      else:
        raise ValueError("Unkown integer")
    elif isinstance(dtype, types.Fixed):
      bits = dtype.bits
      fracs = dtype.fracs
      assert not bits is None, "Must provide bits for a fixed point"
      if fracs is None:
        return "int" + str(bits)
      else:
        assert fracs <= bits, "Fractional part cannot be greater than total bits"
        return "fixed" + str(bits) + "_" + str(fracs)
    elif isinstance(dtype, types.UFixed):
      bits = dtype.bits
      fracs = dtype.fracs
      assert not bits is None, "Must provide bits for a fixed point"
      if fracs is None:
        return "uint" + str(bits)
      else:
        assert fracs <= bits, "Fractional part cannot be greater than total bits"
        return "ufixed" + str(bits) + "_" + str(fracs)

    else:
      raise NotImplementedError()
  else:
    return dtype

def set_name(default, name):
  global NID
  name = default + str(NID) if name is None else name
  NID += 1
  return name

class HCLError(Exception):

  def __init__(self, msg, info):
    frame, filename, line_number, function_name, lines, index = info
    msg = "\"" + filename + "\":" + str(line_number) + " " + msg
    Exception.__init__(self, msg)
  pass

def hcl_excepthook(etype, value, tb):
  if issubclass(etype, HCLError):
    print "[HeteroCL Error] " + value.message
  else:
    sys.__excepthook__(etype, value, tb)
