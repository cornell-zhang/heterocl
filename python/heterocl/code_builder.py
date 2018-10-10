from . import util
from .resizer import CastRemover
from tvm import make as _make
from tvm import stmt as _stmt
from tvm.ir_builder import WithScope
from tvm.api import var as _var, _IterVar
from tvm import ir_pass as _pass

def _pop_stmt(cb):
  stmts = cb.stmt_stack[-1].pop()
  if not stmts or callable(stmts[-1]):
    stmts.append(_make.Evaluate(0))
  stmt = stmts[-1]
  for s in reversed(stmts[:-1]):
    if callable(s):
      stmt = s(stmt)
    else:
      assert isinstance(s, _stmt.Stmt)
      stmt = _make.Block(s, stmt)
  return stmt

class CodeBuilder(object):

  current = []
  stmt_stack = []
  var_dict = []
  axis_list = []
  for_ID = 0

  def __init__(self, name = ""):
    CodeBuilder.stmt_stack.append([[]])
    CodeBuilder.var_dict.append({})
    CodeBuilder.axis_list.append([])
    self.has_break = False
    self.in_for = 0
    self.tensors = set([])
    self.lhs = set([])
    self.name = name
    self.last_stages = set([])

  def __enter__(self):
    CodeBuilder.current.append(self)
    return self

  def __exit__(self, ptype, value, trace):
    CodeBuilder.current.pop()

  def pop_stmt(self):
    return _pop_stmt(CodeBuilder)

  def emit(self, stmt):
    if self.has_break:
      raise ValueError("Cannot write statements after break")
    CodeBuilder.stmt_stack[-1][-1].append(stmt)

  def replace_else(self, if_stmt, else_stmt):
    assert isinstance(if_stmt, _stmt.IfThenElse), "Wrong if statement"
    if isinstance(if_stmt.else_case, _stmt.IfThenElse):
      return _make.IfThenElse(if_stmt.condition, if_stmt.then_case,
          self.replace_else(if_stmt.else_case, else_stmt))
    else:
      return _make.IfThenElse(if_stmt.condition, if_stmt.then_case, else_stmt)


  @staticmethod
  def get():
    stmt = _pop_stmt(CodeBuilder)
    CodeBuilder.stmt_stack.pop()
    CodeBuilder.var_dict.pop()
    CodeBuilder.axis_list.pop()
    assert len(CodeBuilder.current) == len(CodeBuilder.stmt_stack), "Incorrect usage of CodeBuilder"
    return stmt

  @staticmethod
  def get_cb():
    return CodeBuilder.current[-1]

  @staticmethod
  def get_var_dict():
    return CodeBuilder.var_dict[-1]

  @staticmethod
  def get_axis():
    return CodeBuilder.axis_list[-1]

  def _if(self, cond):
    CodeBuilder.stmt_stack[-1].append([])
    def _exit_cb():
      stmt = self.pop_stmt()
      self.has_break = False
      self.emit(_make.IfThenElse(cond, stmt, None))
    return WithScope(None, _exit_cb)

  def _else(self):
    prev = CodeBuilder.stmt_stack[-1][-1][-1]
    CodeBuilder.stmt_stack[-1][-1].pop()
    CodeBuilder.stmt_stack[-1].append([])
    def _exit_cb():
      stmt = self.pop_stmt()
      self.has_break = False
      self.emit(self.replace_else(prev, stmt))
    return WithScope(None, _exit_cb)

  def _elif(self, cond):
    prev = CodeBuilder.stmt_stack[-1][-1][-1]
    CodeBuilder.stmt_stack[-1][-1].pop()
    CodeBuilder.stmt_stack[-1].append([])
    def _exit_cb():
      stmt = self.pop_stmt()
      self.has_break = False
      self.emit(self.replace_else(prev, _make.IfThenElse(cond, stmt, None)))
    return WithScope(None, _exit_cb)

  def _for(self, begin, end, step=1, name=None, dtype="int32", for_type="serial"):
    CodeBuilder.stmt_stack[-1].append([])
    extent = (end - begin)/step
    extent = CastRemover().mutate(extent)
    name = "i"+str(CodeBuilder.for_ID) if name is None else name
    iter_var = _IterVar((0, extent), name, 0)
    CodeBuilder.var_dict[-1][name] = iter_var
    CodeBuilder.axis_list[-1].append(iter_var)
    self.in_for += 1
    def _exit_cb():
      if for_type == "serial":
        for_type_id = 0
      elif for_type == "parallel":
        for_type_id = 1
      elif for_type == "vectorize":
        for_type_id = 2
      elif for_type == "unroll":
        for_type_id = 3
      else:
        raise ValueError("Unknown for_type")
      stmt = _make.AttrStmt(iter_var, "loop_scope", iter_var.var, self.pop_stmt())
      self.has_break = False
      self.in_for -= 1
      self.emit(_make.For(iter_var.var, 0, extent, for_type_id, 0, stmt))
    ret_var = _pass.Simplify(iter_var.var * step + begin)
    return WithScope(ret_var, _exit_cb)

  def _while(self, cond):
    CodeBuilder.stmt_stack[-1].append([])
    self.in_for += 1
    def _exit_cb():
      stmt = self.pop_stmt()
      self.has_break = False
      self.in_for -= 1
      self.emit(_make.While(cond, stmt))
    return WithScope(None, _exit_cb)

  def _for_itervar(self, var, for_type_id = 0):
    CodeBuilder.stmt_stack[-1].append([])
    CodeBuilder.var_dict[-1][var.var.name] = var
    CodeBuilder.axis_list[-1].append(var)
    def _exit_cb():
      if isinstance(var, (list, tuple)):
        self.emit(util.make_for(var, self.pop_stmt(), 0))
      else:
        stmt = _make.AttrStmt(var, "loop_scope", var.var, self.pop_stmt())
        self.emit(_make.For(var.var, var.dom.min, var.dom.extent, for_type_id, 0, stmt))
    return WithScope(None, _exit_cb)
