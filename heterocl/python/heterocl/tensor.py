from . import util
from tvm import make as _make
from tvm import var, decl_buffer

class Tensor:
  def __init__(self, tensor, dtype = ""):
    self.tensor = tensor
    self.ndim = tensor.ndim
    self.shape = tensor.shape
    self.dtype = tensor.dtype if dtype == "" else dtype
    self.index_vars = [var("x"+str(n), dtype="int32") for n in range(0, self.ndim)] #useless
    self.buffer_var = decl_buffer(self.shape, self.dtype, tensor.name).data
    self.stages = []

  # A[x, y] RHS
  def __getitem__(self, indices):
    if not isinstance(indices, tuple):
      indices = (indices,)
    if len(indices) < self.ndim:
      raise ValueError("Not yet defined")
    else:
      index = self.calc_index(indices)
      return _make.Load(self.dtype, self.buffer_var, index, util.true)

  # A[x, y] LHS
  def __setitem__(self, indices, expr):
    if not isinstance(indices, tuple):
      indices = (indices,)
    if len(indices) < self.ndim:
      raise ValueError("Not yet defined")
    else:
      index = self.calc_index(indices)
      return _make.Store(self.buffer_var, expr, index, util.true)

  @property
  def tensor(self):
    return self.tensor

  @property
  def ndim(self):
    return self.ndim

  @property
  def shape(self):
    return self.shape

  @property
  def dtype(self):
    return self.dtype

  @property
  def index_vars(self):
    return self.index_vars

  @property
  def stages(self):
    return self.stages

  def add_stage(self, stage):
    self.stages.append(stage)

  """Helper functions"""
  def calc_index(self, indices):
    index = indices[0]
    for n in range(1, self.ndim):
      index += self.shape[n-1] * indices[n]
    return index
