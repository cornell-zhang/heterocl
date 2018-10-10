from .mutator import IRMutator
import tvm.make as _make
import tvm.expr as _expr
import tvm.stmt as _stmt
from tvm.api import select

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

  def enter_cb(self, cb):
    num = len(cb.stmt_stack[-1])
    for i in range(0, num):
      stmt = cb.stmt_stack[-1][i]
      cb.stmt_stack[-1][i] = self.mutate(stmt)

  def mutate_Var(self, node):
    var = self.get_buf(node)
    if var is None:
      return node
    return var

  def mutate_Cast(self, node): # Remove all casting
    return self.mutate(node.value)

  def mutate_Call(self, node):
    args = []
    for arg in node.args:
      args.append(self.mutate(arg))
    return _make.Call(args[0].dtype, node.name, args, node.call_type, node.func, node.value_index)

  def mutate_Load(self, node):
    index = node.index
    buf = self.get_buf(node.buffer_var)
    if buf is None:
      return _make.Load(node.dtype, node.buffer_var, index, node.predicate)
    else:
      return _make.Load(self.dtype, buf, index, node.predicate)

  def mutate_Store(self, node):
    dtype = node.value.dtype
    value = self.mutate(node.value)
    index = node.index
    buf = self.get_buf(node.buffer_var)
    if buf is None:
      return _make.Store(node.buffer_var, _make.Cast(dtype, value), index, node.predicate)
    else:
      return _make.Store(buf, _make.Cast(self.dtype, value), index, node.predicate)

  def mutate_AttrStmt(self, node):
    body = self.mutate(node.body)
    value = self.mutate(node.value)
    if node.attr_key == "attach_scope":
      buf = self.get_buf(node.node)
      if buf is not None:
        return _make.AttrStmt(buf, node.attr_key, value, body)
    return _make.AttrStmt(node.node, node.attr_key, value, body)

  def mutate_Function(self, node):
    stmt = _make.Evaluate(1)
    ret = node(stmt)
    if isinstance(ret, _stmt.Allocate):
      buf = self.get_buf(ret.buffer_var)
      if buf is None:
        return node
      else:
        return lambda x: _make.Allocate(buf, self.dtype, ret.shape, util.true(), x)
    return node

  def get_buf(self, buf):
    try:
      index = self.from_buf.index(buf)
    except:
      return None
    return self.to_buf[index]

class Downsizer(IRMutator):

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

class CastRemover(IRMutator):

  def mutate_ConstExpr(self, node):
    return node.value

  def mutate_BinOp(binop):
    def decorator(func):
      def op(self, node):
        a = self.mutate(node.a)
        b = self.mutate(node.b)
        if isinstance(a, _expr.ConstExpr):
          a = a.value
        if isinstance(b, _expr.ConstExpr):
          b = b.value
        return binop(a, b, False)
      return op
    return decorator

  @mutate_BinOp(_make.Add)
  def mutate_Add(self, node):
    pass

  @mutate_BinOp(_make.Sub)
  def mutate_Sub(self, node):
    pass

  @mutate_BinOp(_make.Mul)
  def mutate_Mul(self, node):
    pass

  @mutate_BinOp(_make.Div)
  def mutate_Div(self, node):
    pass

  def mutate_Cast(self, node):
    return self.mutate(node.value)
