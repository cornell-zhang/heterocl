import numpy as np
from .tvm.ndarray import array, cpu
from . import types
from .util import get_dtype

def asarray(arr, dtype = None, ctx = cpu(0)):
    dtype = get_dtype(dtype)
    return array(arr, dtype, ctx)

def cast_np(np_in, dtype):

  def cast(val):
    if isinstance(dtype, (types.Fixed, types.UFixed)):
      bits = dtype.bits
      fracs = dtype.fracs
      val = int(val * (1 << fracs))
      mod = val % (1 << bits)
      val = mod if val >= 0 else mod - (1 << bits)
      val = float(val) / (1 << fracs)
      return val
    elif isinstance(dtype, (types.Int, types.UInt)):
      bits = dtype.bits
      mod = int(val) % (1 << bits)
      val = mod if val >= 0 else mod - (1 << bits)
      return val

  vfunc = np.vectorize(cast)
  return vfunc(np_in)

def pack_np(np_in, dtype_in, dtype_out):

  factor = dtype_out.bits / dtype_in.bits
  fracs = dtype_in.fracs
  shape = np_in.shape
  np_out = []
  signed = True
  if isinstance(dtype_in, (types.UInt, types.UFixed)):
    signed = False
  for i in range (0, shape[0]/factor):
    num = 0
    for j in range(0, factor):
      val = int(np_in[i*factor + j] * (1 << fracs))
      if signed:
        val = val if val >= 0 else val + (1 << dtype_in.bits)
      num += val << (j * dtype_in.bits)
    np_out.append(num)
  return np.array(np_out)

def unpack_np(np_in, dtype_in, dtype_out):

  factor = dtype_in.bits / dtype_out.bits
  fracs = dtype_out.fracs
  shape = np_in.shape
  np_out = []
  signed = True
  if isinstance(dtype_out, (types.UInt, types.UFixed)):
    signed = False
  for i in range(0, shape[0] * factor):
    num = int(np_in[i/factor]) >> (dtype_out.bits * (i%factor))
    num = num % (1 << dtype_out.bits)
    if signed:
      num = num if num < 1 << (dtype_out.bits - 1) else num - (1 << dtype_out.bits)
    np_out.append(float(num) / (1 << fracs))
  return np.array(np_out)
