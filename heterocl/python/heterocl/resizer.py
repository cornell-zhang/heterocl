from .mutator import IRMutator
import tvm.make as _make

class Resizer(IRMutator):

  def __init__(self, from_buf, to_buf, dtype):
    self.from_buf = from_buf
    self.to_buf = to_buf
    self.dtype = dtype

  def enter(self, ops):
    stmts = []
    for o in ops:
      if o.body is None:
        stmts.append(None)
      else:
        stmts.append(self.mutate(o.body))
    return stmts

  def mutate_Var(self, node):
    var = self.get_buf(node)
    if var is None:
      return node
    return var

  def mutate_Cast(self, node): # Remove all casting
    return self.mutate(node.value)

  def mutate_Load(self, node):
    index = self.mutate(node.index)
    buf = self.get_buf(node.buffer_var)
    if buf is None:
      return _make.Load(node.dtype, node.buffer_var, index, node.predicate)
    else:
      return _make.Load(self.dtype, buf, index, node.predicate)

  def mutate_Store(self, node):
    dtype = node.value.dtype
    value = self.mutate(node.value)
    index = self.mutate(node.index)
    buf = self.get_buf(node.buffer_var)
    if buf is None:
      return _make.Store(node.buffer_var, _make.Cast(dtype, value), index, node.predicate)
    else:
      return _make.Store(buf, _make.Cast(self.dtype, value), index, node.predicate)

  def get_buf(self, buf):
    try:
      index = self.from_buf.index(buf)
    except:
      return None
    return self.to_buf[index]

