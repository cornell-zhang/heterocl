from . import util
from .code_builder import CodeBuilder
from tvm import make as _make
from tvm import expr as _expr
from tvm.api import decl_buffer
from tvm._ffi.node import NodeGeneric

class TensorSlice(NodeGeneric, _expr.ExprOp):
  def __init__(self, tensor, indices):
    if not isinstance(indices, tuple):
      indices = (indices,)
    self.tensor = tensor
    self.indices = indices

  def __getitem__(self, indices):
    if not isinstance(indices, tuple):
      indices = (indices,)
    return TensorSlice(self.tensor, self.indices + indices)

  def __setitem__(self, indices, expr):
    if not isinstance(indices, tuple):
      indices = (indices,)
    indices = self.indices + indices
    index, bit, _ = util.get_index(self.tensor.shape, indices, 0)
    assert CodeBuilder.current is not None
    builder = CodeBuilder.current
    if bit is None:
      builder.emit(_make.Store(self.tensor.buf.data, _make.Cast(self.tensor.dtype, expr), index, util.true))
    else:
      raise NotImplementedError()

  def asnode(self):
    if len(self.indices) < len(self.tensor.shape):
      raise ValueError("Inconsistant tensor slice dimension with tensor dimension")
    index, bit, _ = util.get_index(self.tensor.shape, self.indices, 0)
    if bit is None:
      return _make.Load(self.tensor.dtype, self.tensor.buf.data, index, util.true)
    else:
      return _make.GetBit(_make.Load(self.tensor.dtype, self.tensor.buf.data, index, util.true), bit)

  @property
  def dtype(self):
    return self.tensor.dtype


# A wrapper for TVM tensor
class Tensor(NodeGeneric, _expr.ExprOp):
  def __init__(self, tensor, buf = None):
    self._tensor = tensor
    self._buf = buf
    if buf is None:
      self._buf = decl_buffer(tensor.shape, tensor.dtype, tensor.name)

  # A[x, y] RHS
  def __getitem__(self, indices):
    if not isinstance(indices, tuple):
      indices = (indices,)
    return TensorSlice(self, indices)

  # A[x, y] LHS
  def __setitem__(self, indices, expr):
    if not isinstance(indices, tuple):
      indices = (indices,)
    if len(indices) < self.tensor.ndim:
      raise NotImplementedError()
    else:
      index, bit, _ = util.get_index(self.tensor.shape, indices, 0)
      assert CodeBuilder.current is not None
      builder = CodeBuilder.current
      if bit is None:
        builder.emit(_make.Store(self.buf.data, _make.Cast(self.tensor.dtype, expr), index, util.true))
      else:
        raise NotImplementedError()

  def asnode(self):
    return TensorSlice(self, 0).asnode()

  @property
  def tensor(self):
    return self._tensor

  @property
  def buf(self):
    return self._buf

  @property
  def shape(self):
    return self.tensor.shape

  @property
  def dtype(self):
    return self.tensor.dtype

  @property
  def op(self):
    return self.tensor.op

  @buf.setter
  def buf(self, buf):
    self._buf = buf
