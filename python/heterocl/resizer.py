from .mutator import IRMutator
import tvm.make as _make
import tvm.expr as _expr
import tvm.stmt as _stmt
from tvm.api import select

class Quantizer(IRMutator):

  def __init__(self, from_buf, dt_var):
    self.from_buf = from_buf
    self.dt_var = dt_var

  def enter(self, ops):
    stmts = []
    for o in ops:
      if o.body is None:
        stmts.append(None)
      else:
        stmts.append(self.mutate(o.body))
    return stmts

  def mutate_Var(self, node):
    if node in self.from_buf:
      return _make.Quantize(node, self.dt_var)
    else:
      return node

  def mutate_Cast(self, node):
    return self.mutate(node.value)

  def mutate_Add(self, node):
    a = self.mutate(node.a)
    b = self.mutate(node.b)
    if not isinstance(a, _expr.Quantize) and not isinstance(b, _expr.Quantize):
      return _make.Add(a, b)
    sa, a_bw = self.get_bits(a)
    sb, b_bw = self.get_bits(b)
    so, o_bw = self.get_bits(node)
    if isinstance(a, _expr.Quantize):
      a_bw = a.bitwidth
    if isinstance(b, _expr.Quantize):
      b_bw = b.bitwidth
    n_bw = _make.Add(select(a_bw > b_bw, a_bw, b_bw), 1, False) # do not cast
    if sa != sb:
      n_bw = _make.Add(n_bw, 1, False) # do not cast
    if n_bw == o_bw:
      return node
    return _make.Quantize(_make.Add(a, b), n_bw)

  def mutate_Sub(self, node):
    a = self.mutate(node.a)
    b = self.mutate(node.b)
    if not isinstance(a, _expr.Quantize) and not isinstance(b, _expr.Quantize):
      return _make.Sub(a, b)
    sa, a_bw = self.get_bits(node.a)
    sb, b_bw = self.get_bits(node.b)
    so, o_bw = self.get_bits(node)
    if isinstance(a, _expr.Quantize):
      a_bw = a.bitwidth
    if isinstance(b, _expr.Quantize):
      b_bw = b.bitwidth
    n_bw = _make.Add(select(a_bw > b_bw, a_bw, b_bw), 1, False) # do not cast
    if sa != sb:
      n_bw = _make.Add(n_bw, 1, False) # do not cast
    if n_bw == o_bw:
      return node
    return _make.Quantize(_make.Sub(a, b), n_bw)

  def mutate_Mul(self, node):
    a = self.mutate(node.a)
    b = self.mutate(node.b)
    if not isinstance(a, _expr.Quantize) and not isinstance(b, _expr.Quantize):
      return _make.Mul(a, b)
    sa, a_bw = self.get_bits(node.a)
    sb, b_bw = self.get_bits(node.b)
    so, o_bw = self.get_bits(node)
    if isinstance(a, _expr.Quantize):
      a_bw = a.bitwidth
    if isinstance(b, _expr.Quantize):
      b_bw = b.bitwidth
    n_bw = _make.Add(a_bw, b_bw, False) # do not cast
    if n_bw == o_bw:
      return node
    return _make.Quantize(_make.Add(a, b), n_bw)

  def mutate_Div(self, node):
    a = self.mutate(node.a)
    b = self.mutate(node.b)
    if not isinstance(a, _expr.Quantize) and not isinstance(b, _expr.Quantize):
      return _make.Div(a, b)
    sa, a_bw = self.get_bits(node.a)
    sb, b_bw = self.get_bits(node.b)
    so, o_bw = self.get_bits(node)
    if isinstance(a, _expr.Quantize):
      a_bw = a.bitwidth
    if isinstance(b, _expr.Quantize):
      b_bw = b.bitwidth
    if sb: # signed divisor
      n_bw = _make.Add(a_bw, 1, False)
    else:
      n_bw = a_bw
    if n_bw == o_bw:
      return node
    return _make.Quantize(_make.Add(a, b), n_bw)

  def mutate_Load(self, node):
    index = node.index
    load = _make.Load(node.dtype, node.buffer_var, index, node.predicate)
    if node.buffer_var not in self.from_buf:
      return load
    return _make.Quantize(load, self.dt_var)

  def mutate_Store(self, node):
    dtype = node.value.dtype
    value = self.mutate(node.value)
    index = node.index
    if node.buffer_var not in self.from_buf:
      return _make.Store(node.buffer_var, _make.Cast(dtype, value), index, node.predicate)
    return _make.Store(node.buffer_var, _make.Cast(dtype, _make.Quantize(value, self.dt_var)), index, node.predicate)

  def mutate_GetBit(self, node):
    a = self.mutate(node.a)
    gb = _make.GetBit(a, node.index)
    if isinstance(a, _expr.Quantize):
      return _make.Quantize(gb, self.dt_var)
    return gb

  def get_bits(self, expr):
    dtype = expr.dtype
    if dtype[0:3] == "int":
      return True, int(dtype[3:])
    elif dtype[0:4] == "uint":
      return False, int(dtype[4:])
    else:
      raise ValueError("Cannot perform down sampling with non-integer type")


