"""A module for HeteroCL imperative code support

We use a stack to store all the imperative statements. The stack is a
three-dimensional array. The first dimension represents the current
API scope. The second dimension is the current with scope. The third
dimension is the statement itself.
"""
from .tvm import make as _make
from .tvm import stmt as _stmt
from .tvm.ir_builder import WithScope
from .tvm.api import _IterVar
from .tvm import ir_pass as _pass
from .resizer import CastRemover
from . import util

def _pop_stmt(cb):
    """Collect all statements under a CodeBuilder and combine them into
    a single statment.
    """
    stmts = cb.get_stmt_stack().pop()
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
    """Basic builder for mixed-imperative-declarative programming.

    CodeBuilder is a class that

    Attributes
    ----------
    current : list
        Store all alive CodeBuilder. The newest is at the end.

    stmt_stack : list
        Store all statments. There are three levels. The outer-most
        level is for different CodeBuilder. The second level is for
        different scopes of statement. The inner-most level is for
        different statements.

    var_dict : list
        A list of dictionaries whose key is the name of the variable
        and the value is the variable itself. This enables users to
        access a variable inside a CodeBuilder via a Python attribute.

    axis_list : list
        A list of lists of axes.

    """
    current = []
    for_ID = 0

    def __init__(self, name = ""):
        self.stmt_stack = [[]]
        self.var_dict = {}
        self.axis_list = []
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
        pass
        #CodeBuilder.current.pop()

    def pop_stmt(self):
        return _pop_stmt(CodeBuilder)

    def emit(self, stmt):
        if self.has_break:
            raise ValueError("Cannot write statements after break")
        CodeBuilder.get_stmt_stack()[-1].append(stmt)

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
        CodeBuilder.current.pop()
        #assert len(CodeBuilder.current) == len(CodeBuilder.stmt_stack), "Incorrect usage of CodeBuilder"
        return stmt

    @staticmethod
    def get_cb():
        return CodeBuilder.current[-1]

    @staticmethod
    def get_stmt_stack():
        return CodeBuilder.current[-1].stmt_stack

    @staticmethod
    def get_var_dict():
        return CodeBuilder.current[-1].var_dict

    @staticmethod
    def get_axis_list():
        return CodeBuilder.current[-1].axis_list

    @staticmethod
    def get_len():
        return len(CodeBuilder.current)

    def _if(self, cond):
        CodeBuilder.get_stmt_stack().append([])
        def _exit_cb():
            stmt = self.pop_stmt()
            self.has_break = False
            self.emit(_make.IfThenElse(cond, stmt, None))
        return WithScope(None, _exit_cb)

    def _else(self):
        prev = CodeBuilder.get_stmt_stack()[-1][-1]
        CodeBuilder.get_stmt_stack()[-1].pop()
        CodeBuilder.get_stmt_stack().append([])
        def _exit_cb():
            stmt = self.pop_stmt()
            self.has_break = False
            self.emit(self.replace_else(prev, stmt))
        return WithScope(None, _exit_cb)

    def _elif(self, cond):
        prev = CodeBuilder.get_stmt_stack()[-1][-1]
        CodeBuilder.get_stmt_stack()[-1].pop()
        CodeBuilder.get_stmt_stack().append([])
        def _exit_cb():
            stmt = self.pop_stmt()
            self.has_break = False
            self.emit(self.replace_else(prev, _make.IfThenElse(cond, stmt, None)))
        return WithScope(None, _exit_cb)

    def _for(self, begin, end, step=1, name=None, dtype="int32", for_type="serial"):
        CodeBuilder.get_stmt_stack().append([])
        extent = (end - begin)/step
        extent = CastRemover().mutate(extent)
        name = "i"+str(CodeBuilder.for_ID) if name is None else name
        iter_var = _IterVar((0, extent), name, 0)
        CodeBuilder.get_var_dict()[name] = iter_var
        CodeBuilder.get_axis_list().append(iter_var)
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
        CodeBuilder.get_stmt_stack().append([])
        self.in_for += 1
        def _exit_cb():
            stmt = self.pop_stmt()
            self.has_break = False
            self.in_for -= 1
            self.emit(_make.While(cond, stmt))
        return WithScope(None, _exit_cb)

    def _for_itervar(self, var, for_type_id = 0):
        CodeBuilder.get_stmt_stack().append([])
        CodeBuilder.get_var_dict()[var.var.name] = var
        CodeBuilder.get_axis_list().append(var)
        def _exit_cb():
            if isinstance(var, (list, tuple)):
                self.emit(util.make_for(var, self.pop_stmt(), 0))
            else:
                stmt = _make.AttrStmt(var, "loop_scope", var.var, self.pop_stmt())
                self.emit(_make.For(var.var, var.dom.min, var.dom.extent, for_type_id, 0, stmt))
        return WithScope(None, _exit_cb)
