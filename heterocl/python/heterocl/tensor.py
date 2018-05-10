from . import util
from .resizer import CastRemover
from .code_builder import CodeBuilder
from tvm import make as _make
from tvm import expr as _expr
from tvm.api import decl_buffer
from tvm._ffi.node import NodeGeneric

class Var(NodeGeneric, _expr.ExprOp):
  def __init__(self, var):
    self._var = var

  @property
  def var(self):
    return self._var

  @property
  def name(self):
    return self._var.name

  @property
  def dtype(self):
    return self._var.dtype

  @var.setter
  def var(self, var):
    self._var = var

  def asnode(self):
    return self._var

class TensorSlice(NodeGeneric, _expr.ExprOp):
  def __init__(self, tensor, indices):
    if not isinstance(indices, tuple):
      indices = (indices,)
    self.tensor = tensor
    self.indices = indices

  def __getitem__(self, indices):
    indices = CastRemover().mutate(indices)
    if not isinstance(indices, tuple):
      indices = (indices,)
    return TensorSlice(self.tensor, self.indices + indices)

  def __setitem__(self, indices, expr):
    if not isinstance(indices, tuple):
      indices = (indices,)
    indices = self.indices + indices
    index, bit, _ = util.get_index(self.tensor.shape, indices, 0)
    builders = CodeBuilder.current
    assert len(builders) != 0
    builder = builders[-1]
    if bit is None:
      builder.emit(_make.Store(self.tensor.buf.data, _make.Cast(self.tensor.dtype, expr), index))
    elif type(bit) == slice:
      load = _make.Load(self.tensor.dtype, self.tensor.buf.data, index)
      expr = _make.SetSlice(load, expr, bit.start, bit.stop)
      builder.emit(_make.Store(self.tensor.buf.data, _make.Cast(self.tensor.dtype, expr), index))
    else:
      load = _make.Load(self.tensor.dtype, self.tensor.buf.data, index)
      expr = _make.SetBit(load, expr, bit)
      builder.emit(_make.Store(self.tensor.buf.data, _make.Cast(self.tensor.dtype, expr), index))

  def asnode(self):
    if len(self.indices) < len(self.tensor.shape):
      raise ValueError("Inconsistant tensor slice dimension with tensor dimension")
    index, bit, _ = util.get_index(self.tensor.shape, self.indices, 0)
    if bit is None:
      return _make.Load(self.tensor.dtype, self.tensor.buf.data, index)
    elif type(bit) == slice:
      return _make.GetSlice(_make.Load(self.tensor.dtype, self.tensor.buf.data, index), bit.start, bit.stop)
    else:
      return _make.GetBit(_make.Load(self.tensor.dtype, self.tensor.buf.data, index), bit)

  @property
  def dtype(self):
    return self.tensor.dtype


# A wrapper for TVM tensor
class Tensor(NodeGeneric, _expr.ExprOp):

  tensor_map = {}

  def __init__(self, shape, dtype = "int32", name = "hcl.tensor", buf = None):
    self._tensor = None
    self._buf = buf
    self._dtype = dtype
    self._shape = shape
    self.name = name
    if buf is None:
      self._buf = decl_buffer(shape, dtype, name)

  # A[x, y] RHS
  def __getitem__(self, indices):
    indices = CastRemover().mutate(indices)
    if not isinstance(indices, tuple):
      indices = (indices,)
    return TensorSlice(self, indices)

  # A[x, y] LHS
  def __setitem__(self, indices, expr):
    if not isinstance(indices, tuple):
      indices = (indices,)
    if len(indices) < len(self._shape):
      raise NotImplementedError()
    else:
      index, bit, _ = util.get_index(self._shape, indices, 0)
      builders = CodeBuilder.current
      assert len(builders) != 0
      if bit is None:
        builders[-1].emit(_make.Store(self.buf.data, _make.Cast(self._dtype, expr), index))
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
    return self._shape

  @property
  def dtype(self):
    return self._dtype

  @property
  def op(self):
    return self.tensor.op

  @buf.setter
  def buf(self, buf):
    self._buf = buf
    Tensor.tensor_map[self._tensor] = buf

  @dtype.setter
  def dtype(self, dtype):
    self._dtype = dtype

  @tensor.setter
  def tensor(self, tensor):
    self._tensor = tensor

class Operation():

  op_list = []

  def __init__(self, inputs, output, body):
    self.inputs = inputs
    self.output = output
    self.body = body

  @property
  def inputs(self):
    return self.inputs

  @property
  def output(self):
    return self.output

  @property
  def body(self):
    return self.body

  @body.setter
  def body(self, body):
    self.body = body
