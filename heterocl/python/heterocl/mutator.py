from tvm import expr as _expr
from tvm import stmt as _stmt
from tvm import make as _make

class API():

  def __init__(self, args, _type):
    self.args = args
    self._type = _type

  @property
  def args(self):
    return self.args

  @property
  def _type(self):
    return self._type


class IRMutator(object):

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
    elif isinstance(node, _stmt.Stmt):
      if isinstance(node, _stmt.LetStmt):
        return self.mutate_LetStmt(node)
      elif isinstance(node, _stmt.AssertStmt):
        return self.mutate_AssertStmt(node)
      elif isinstance(node, _stmt.ProducerConsumer):
        return self.mutate_ProducerConsumer(node)
      elif isinstance(node, _stmt.For):
        return self.mutate_For(node)
      elif isinstance(node, _stmt.Store):
        return self.mutate_Store(node)
      elif isinstance(node, _stmt.Provide):
        return self.mutate_Provide(node)
      elif isinstance(node, _stmt.Allocate):
        return self.mutate_Allocate(node)
      elif isinstance(node, _stmt.AttrStmt):
        return self.mutate_AttrStmt(node)
      elif isinstance(node, _stmt.Free):
        return self.mutate_Free(node)
      elif isinstance(node, _stmt.Realize):
        return self.mutate_Realize(node)
      elif isinstance(node, _stmt.Block):
        return self.mutate_Block(node)
      elif isinstance(node, _stmt.IfThenElse):
        return self.mutate_IfThenElse(node)
      elif isinstance(node, _stmt.Evaluate):
        return self.mutate_Evaluate(node)
      elif isinstance(node, _stmt.Prefetch):
        return self.mutate_Prefetch(node)
      else:
        return node
    elif isinstance(node, tuple):
      return self.mutate_Tuple(node)
    elif isinstance(node, list):
      return self.mutate_List(node)
    elif isinstance(node, API):
      return self.mutate_API(node)
    else:
      return node

  def mutate_BinOp(binop):
    def decorator(func):
      def op(self, node):
        a = self.mutate(node.a)
        b = self.mutate(node.b)
        return binop(a, b)
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

  @mutate_BinOp(_make.Mod)
  def mutate_Mod(self, node):
    pass

  @mutate_BinOp(_make.Min)
  def mutate_Min(self, node):
    pass

  @mutate_BinOp(_make.Max)
  def mutate_Max(self, node):
    pass

  @mutate_BinOp(_make.EQ)
  def mutate_EQ(self, node):
    pass

  @mutate_BinOp(_make.NE)
  def mutate_NE(self, node):
    pass

  @mutate_BinOp(_make.LT)
  def mutate_LT(self, node):
    pass

  @mutate_BinOp(_make.LE)
  def mutate_LE(self, node):
    pass

  @mutate_BinOp(_make.GT)
  def mutate_GT(self, node):
    pass

  @mutate_BinOp(_make.GE)
  def mutate_GE(self, node):
    pass

  def mutate_Call(self, node):
    args = []
    for arg in node.args:
      args.append(self.mutate(arg))
    return _make.Call(node.dtype, node.name, args, node.call_type, node.func, node.value_index)

  def mutate_Var(self, node):
    return node

  def mutate_Cast(self, node):
    value = self.mutate(node.value)
    return _make.Cast(node.dtype, value)

  def mutate_Load(self, node):
    buffer_var = self.mutate(node.buffer_var)
    index = self.mutate(node.index)
    predicate = self.mutate(node.predicate)
    return _make.Load(node.dtype, buffer_var, index, predicate)

  def mutate_For(self, node):
    loop_var = self.mutate(node.loop_var)
    _min = self.mutate(node.min)
    extent = self.mutate(node.extent)
    body = self.mutate(node.body)
    return _make.For(loop_var, _min, extent, node.for_type, node.device_api, body)

  def mutate_Store(self, node):
    buffer_var = self.mutate(node.buffer_var)
    index = self.mutate(node.index)
    value = self.mutate(node.value)
    predicate = self.mutate(node.predicate)
    return _make.Store(buffer_var, value, index, predicate)

  def mutate_IfThenElse(self, node):
    condition = self.mutate(node.condition)
    then_case = self.mutate(node.then_case)
    else_case = self.mutate(node.else_case)
    return _make.IfThenElse(condition, then_case, else_case)

  def mutate_Tuple(self, node):
    _list = list(node)
    self.mutate(_list)
    return tuple(_list)

  def mutate_List(self, node):
    _len = len(node)
    _list = []
    for i in range(0, _len):
      _list.append(self.mutate(node[i]))
    return _list

  def mutate_API(self, node):
    args = self.mutate(args)
    return API(args, node._type)
