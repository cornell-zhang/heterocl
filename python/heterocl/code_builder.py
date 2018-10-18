"""A module for HeteroCL imperative code support

We use a stack to store all the imperative statements. The stack is a
three-dimensional array. The first dimension represents the _current
API scope. The second dimension is the _current with scope. The third
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

    CodeBuilder is a class that help build an imperative code block.
    Thus class is mainly for internal use. However, users can use this
    class for debugging. A CodeBuilder should be used with a `with`
    statement. A :code:`CodeBuilder.get()` must be used after a block
    of CodeBuilder.

    Examples
    --------
    .. code-block:: python

        # following shows an example of using CodeBuilder
        with hcl.CodeBuilder() as cb:
            A = hcl.compute((10,), lambda x: 0)
            with hcl.for_(0, 9) as i:
                A[i] = A[i+1] - 1
        # get the statements inside the CodeBuilder
        stmt = CodeBuilder.get()

    Parameters
    ----------
    name : str, optional
        The name of the CodeBuilder.

    Attributes
    ----------
    stmt_stack : :obj:`list` of :obj:`list` of :class:`.Stmt`
        Store all statments. There are two levels. The outer level is
        for different scopes of statement. The inner level is for
        different statements.

    var_dict : :obj:`dict` of :obj:`str`:
        A dictionary whose key is the name of the variable
        and the value is the variable itself. This enables users to
        access a variable inside a CodeBuilder via a Python attribute.

    axis_list : :obj:`list`
        A list of axes appeared in this CodeBuilder.

    has_break : bool
        Set to `True` if there is a `break` statement inside a `for` or
        `while` loop.

    for_level : int
        The level of a loop nest where the current statement is.

    """
    _current = []
    """Store all alive CodeBuilder. The newest is at the end."""

    def __init__(self, name = ""):
        self.stmt_stack = [[]]
        self.var_dict = {}
        self.axis_list = []
        self.has_break = False
        self.for_level = 0
        self.tensors = set([])
        self.lhs = set([])
        self.name = name
        self.last_stages = set([])
        self.for_ID = 0

    def __enter__(self):
        CodeBuilder._current.append(self)
        return self

    def __exit__(self, ptype, value, trace):
        pass
        #CodeBuilder._current.pop()

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
        CodeBuilder._current.pop()
        return stmt

    @staticmethod
    def get_cb():
        return CodeBuilder._current[-1]

    @staticmethod
    def get_stmt_stack():
        return CodeBuilder._current[-1].stmt_stack

    @staticmethod
    def get_var_dict():
        return CodeBuilder._current[-1].var_dict

    @staticmethod
    def get_axis_list():
        return CodeBuilder._current[-1].axis_list

    @staticmethod
    def get_len():
        return len(CodeBuilder._current)

    def _if(self, cond):
        cb = CodeBuilder.get_cb()
        cb.stmt_stack.append([])
        def _exit_cb():
            stmt = self.pop_stmt()
            self.has_break = False
            self.emit(_make.IfThenElse(cond, stmt, None))
        return WithScope(None, _exit_cb)

    def _else(self):
        cb = CodeBuilder.get_cb()
        prev = cb.stmt_stack[-1][-1]
        cb.stmt_stack[-1].pop()
        cb.stmt_stack.append([])
        def _exit_cb():
            stmt = self.pop_stmt()
            self.has_break = False
            self.emit(self.replace_else(prev, stmt))
        return WithScope(None, _exit_cb)

    def _elif(self, cond):
        cb = CodeBuilder.get_cb()
        prev = cb.stmt_stack[-1][-1]
        cb.stmt_stack[-1].pop()
        cb.stmt_stack.append([])
        def _exit_cb():
            stmt = self.pop_stmt()
            self.has_break = False
            self.emit(self.replace_else(prev, _make.IfThenElse(cond, stmt, None)))
        return WithScope(None, _exit_cb)

    def _for(self, begin, end, step=1, name=None, dtype="int32", for_type="serial"):
        cb = CodeBuilder.get_cb()
        cb.stmt_stack.append([])
        extent = (end - begin)/step
        extent = CastRemover().mutate(extent)
        name = "i"+str(cb.for_ID) if name is None else name
        cb.for_ID += 1
        iter_var = _IterVar((0, extent), name, 0)
        cb.var_dict[name] = iter_var
        cb.axis_list.append(iter_var)
        self.for_level += 1
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
            self.for_level -= 1
            self.emit(_make.For(iter_var.var, 0, extent, for_type_id, 0, stmt))
        ret_var = _pass.Simplify(iter_var.var * step + begin)
        return WithScope(ret_var, _exit_cb)

    def _while(self, cond):
        cb = CodeBuilder.get_cb()
        cb.stmt_stack.append([])
        self.for_level += 1
        def _exit_cb():
            stmt = self.pop_stmt()
            self.has_break = False
            self.for_level -= 1
            self.emit(_make.While(cond, stmt))
        return WithScope(None, _exit_cb)

    def _for_itervar(self, var, for_type_id = 0):
        cb = CodeBuilder.get_cb()
        cb.stmt_stack().append([])
        cb.var_dict[var.var.name] = var
        cb.axis_list.append(var)
        def _exit_cb():
            if isinstance(var, (list, tuple)):
                self.emit(util.make_for(var, self.pop_stmt(), 0))
            else:
                stmt = _make.AttrStmt(var, "loop_scope", var.var, self.pop_stmt())
                self.emit(_make.For(var.var, var.dom.min, var.dom.extent, for_type_id, 0, stmt))
        return WithScope(None, _exit_cb)
