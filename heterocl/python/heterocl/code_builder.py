from tvm import make as _make
from tvm import stmt as _stmt
from tvm.ir_builder import WithScope
from tvm.api import var as _var
from tvm import ir_pass as _pass

def _pop_stmt(cb):
  stmts = cb.stmt_stack.pop()
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

  current = None
  stmt_stack = None

  def __init__(self):
    CodeBuilder.stmt_stack = [[]]

  def __enter__(self):
    CodeBuilder.current = self
    return self

  def __exit__(self, ptype, value, trace):
    CodeBuilder.current = None

  def pop_stmt(self):
    return _pop_stmt(CodeBuilder)

  def emit(self, stmt):
    CodeBuilder.stmt_stack[-1].append(stmt)

  @staticmethod
  def get():
    stmt = _pop_stmt(CodeBuilder)
    CodeBuilder.stmt_stack = None
    return stmt

  def _if(self, cond):
    CodeBuilder.stmt_stack.append([])
    def _exit_cb():
      self.emit(_make.IfThenElse(cond, self.pop_stmt(), None))
    return WithScope(None, _exit_cb)

  def _else(self):
    prev = CodeBuilder.stmt_stack[-1][-1]
    CodeBuilder.stmt_stack[-1].pop()
    CodeBuilder.stmt_stack.append([])
    def _exit_cb():
      self.emit(_make.IfThenElse(prev.condition, prev.then_case, self.pop_stmt()))
    return WithScope(None, _exit_cb)

  def _for(self, begin, end, name="i", dtype="int32", for_type="serial"):
    CodeBuilder.stmt_stack.append([])
    loop_var = _var(name, dtype=dtype)
    extent = end if begin == 0 else _pass.Simplify(end - begin)
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
      self.emit(_make.For(loop_var, begin, extent, for_type_id, 0, self.pop_stmt()))
    return WithScope(loop_var, _exit_cb)
