from tvm import expr as _expr
from tvm import stmt as _stmt
from tvm import make as _make

class IRMutator():

  def mutate(self, node):
    if isinstance(node, _expr.Expr):
      if isinstance(node, _expr.ConstExpr):
        return node
      elif isinstance(node, _expr.BinaryOpExpr):
        if isinstance(node, _expr.Add):
          return self.mutate_Add(node)
        elif isinstance(node, _expr.Sub):
          return self.mutate_Sub(node)
        elif isinstance(node, _expr.Mul):
          return self.mutate_Mul(node)
        elif isinstance(node, _expr.Div):
          return self.mutate_Div(node)
        elif isinstance(node, _expr.Mod):
          return self.mutate_Mod(node)
        elif isinstance(node, _expr.Min):
          return self.mutate_Min(node)
        elif isinstance(node, _expr.Max):
          return self.mutate_Max(node)
        else:
          return node
      elif isinstance(node, _expr.CmpExpr):
        if isinstance(node, _expr.EQ):
          return self.mutate_EQ(node)
        elif isinstance(node, _expr.NE):
          return self.mutate_NE(node)
        elif isinstance(node, _expr.LT):
          return self.mutate_LT(node)
        elif isinstance(node, _expr.LE):
          return self.mutate_LE(node)
        elif isinstance(node, _expr.GT):
          return self.mutate_GT(node)
        elif isinstance(node, _expr.GE):
          return self.mutate_GE(node)
        else:
          return node
      elif isinstance(node, _expr.LogicalExpr):
        if isinstance(node, _expr.And):
          return self.mutate_And(node)
        elif isinstance(node, _expr.Or):
          return self.mutate_Or(node)
        elif isinstance(node, _expr.Not):
          return self.mutate_Not(node)
        else:
          return node
      else:
        if isinstance(node, _expr.Var):
          return self.mutate_Var(node)
        elif isinstance(node, _expr.Reduce):
          return self.mutate_Reduce(node)
        elif isinstance(node, _expr.Cast):
          return self.mutate_Cast(node)
        elif isinstance(node, _expr.Select):
          return self.mutate_Select(node)
        elif isinstance(node, _expr.Load):
          return self.mutate_Load(node)
        elif isinstance(node, _expr.Call):
          return self.mutate_Call(node)
        elif isinstance(node, _expr.Let):
          return self.mutate_Let(node)
        else:
          return node
    else:
      if isinstance(node, _stmt.Store):
        return self.mutate_Store(node)
      elif isinstance(node, _stmt.For):
        return self.mutate_For(node)
      else:
        return node

  def mutate_Add(self, node):
    a = self.mutate(node.a)
    b = self.mutate(node.b)
    return _make.Add(a, b)

  def mutate_Mul(self, node):
    a = self.mutate(node.a)
    b = self.mutate(node.b)
    return _make.Mul(a, b)

  def mutate_Call(self, node):
    args = []
    for arg in node.args:
      args.append(self.mutate(arg))
    return _make.Call(node.dtype, node.name, args, node.call_type, node.func, node.value_index)

  def mutate_Var(self, node):
    return node

  def mutate_Load(self, node):
    buffer_var = self.mutate(node.buffer_var)
    index = self.mutate(node.index)
    predicate = self.mutate(node.predicate)
    return _make.Load(node.dtype, buffer_var, index, predicate)

  def mutate_Store(self, node):
    buffer_var = self.mutate(node.buffer_var)
    index = self.mutate(node.index)
    value = self.mutate(node.value)
    predicate = self.mutate(node.predicate)
    return _make.Store(buffer_var, value, index, predicate)

  def mutate_For(self, node):
    loop_var = self.mutate(node.loop_var)
    _min = self.mutate(node.min)
    extent = self.mutate(node.extent)
    body = self.mutate(node.body)
    return _make.For(loop_var, _min, extent, node.for_type, node.device_api, body)
