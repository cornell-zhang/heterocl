"""A module for HeteroCL imperative code support"""
from .tvm import make as _make
from .tvm import stmt as _stmt
from .tvm import api as tvm_api
from .tvm.ir_builder import WithScope
from .tvm.api import _IterVar
from .tvm._api_internal import _ExternOp
from .tvm import ir_pass as _pass
from .resizer import CastRemover
from . import util
from .schedule import Schedule

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

class Stage(object):
    """Basic builder for mixed-imperative-declarative programming.

    Stage is a class that help build an imperative code block.
    Thus class is mainly for internal use. However, users can use this
    class for debugging. A Stage should be used with a `with`
    statement. A :code:`Stage.get()` must be used after a block
    of Stage. A block formed within a Stage will be a
    :class:`.Stage` that has input stages

    Examples
    --------
    .. code-block:: python

        # following shows an example of using Stage
        with hcl.Stage() as cb:
            A = hcl.compute((10,), lambda x: 0)
            with hcl.for_(0, 9) as i:
                A[i] = A[i+1] - 1
        # get the statements inside the Stage
        stmt = Stage.get()

    Parameters
    ----------
    name : str, optional
        The name of the Stage.

    Attributes
    ----------
    stmt_stack : list[list[Stmt]]
        Store all statments. There are two levels. The outer level is
        for different scopes of statement. The inner level is for
        different statements.

    var_dict : dict(str, Var)
        A dictionary whose key is the name of the variable
        and the value is the variable itself. This enables users to
        access a variable inside a Stage via a Python attribute.

    axis_list : list[IterVar]
        A list of axes appeared in this Stage.

    has_break : bool
        Set to `True` if there is a `break` statement inside a `for` or
        `while` loop.

    for_level : int
        The level of a loop nest where the current statement is.

    input_stages(tensors) : set(Stage)
        A set of stages that are the input to the Stage.

    """
    _current = []
    """Store all living `Stage`. The newest is at the end."""

    def __init__(self, name=None, dtype=None, shape=()):
        # Attributes related to a single stage
        self.name = util.get_name("stage", name)
        self.stmt_stack = [[]]
        self.var_dict = {}
        self.axis_list = []
        self.has_break = False
        self.for_level = 0
        self.for_ID = 0
        # Attributes for corss-stage relation
        self.input_stages = set([])
        self.lhs_tensors = set([])
        self.last_substages = set([])
        # Private attributes for buildind a stage
        self._op = None
        self._dtype = util.get_dtype(dtype)
        self._buf = tvm_api.decl_buffer(shape, self._dtype, self.name)
        self._shape = self._buf.shape

    def __enter__(self):
        Stage._current.append(self)
        return self

    def __exit__(self, ptype, value, trace):
        #Stage._current.pop()
        # update input_stages: the union of the last substages and original input stages collected in the stage
        self.input_stages = self.last_substages.union(self.input_stages)
        # create the output operation
        input_ops = [i._op for i in self.input_stages]
        input_bufs = [i._buf for i in self.input_stages]
        output_bufs = [self._buf]
        body = _pop_stmt(Stage) #TODO: need to fix here
        Stage._current.pop()
        op = _ExternOp(self.name, "", self.axis_list, input_ops,
                       input_bufs, output_bufs, body)
        self._op = op.output(0)
        # update last_update stages
        # if this stage is a substage of other stages
        if len(Stage._current) > 0:
            superstage = Stage._current[-1]
            # add attribute statement for later stage insertion
            superstage.emit(
                lambda x: _make.AttrStmt(self._buf, "attach_scope",
                                         _make.StringImm(superstage.name), x))
            # update the last substages of the superstage:
            # last_substages = original substages + current stage - inputs of current stage
            superstage.last_substages.add(self)
            superstage.last_substages.difference_update(self.input_stages)
            # update lhs_tensors:
            # lhs_tensors = original tensors + lhs tensors of current stage
            superstage.lhs_tensors.update(self.lhs_tensors)
            # update var_dict
            superstage.var_dict[self.name] = self
        # Otherwise update the list of stages globally
        else:
            Schedule.stage_ops.append(self)
            Schedule.last_stages.add(self)
            Schedule.last_stages.difference_update(self.input_stages)

    def __repr__(self):
        return self.name

    def __getattr__(self, name):
        try:
            return self.var_dict[name]
        except KeyError:
            raise ValueError("Uknown member " + name + " of " + self.name)

    def pop_stmt(self):
        return _pop_stmt(Stage)

    def emit(self, stmt):
        if self.has_break:
            raise ValueError("Cannot write statements after break")
        Stage.get_stmt_stack()[-1].append(stmt)

    def replace_else(self, if_stmt, else_stmt):
        assert isinstance(if_stmt, _stmt.IfThenElse), "Wrong if statement"
        if isinstance(if_stmt.else_case, _stmt.IfThenElse):
            return _make.IfThenElse(if_stmt.condition, if_stmt.then_case,
                    self.replace_else(if_stmt.else_case, else_stmt))
        else:
            return _make.IfThenElse(if_stmt.condition, if_stmt.then_case, else_stmt)


    @staticmethod
    def get():
        stmt = _pop_stmt(Stage)
        Stage._current.pop()
        return stmt

    @staticmethod
    def get_cb():
        return Stage._current[-1]

    @staticmethod
    def get_stmt_stack():
        return Stage._current[-1].stmt_stack

    @staticmethod
    def get_var_dict():
        return Stage._current[-1].var_dict

    @staticmethod
    def get_axis_list():
        return Stage._current[-1].axis_list

    @staticmethod
    def get_len():
        return len(Stage._current)

    @property
    def axis(self):
        return self._op.op.axis

    def _if(self, cond):
        cb = Stage.get_cb()
        cb.stmt_stack.append([])
        def _exit_cb():
            stmt = self.pop_stmt()
            self.has_break = False
            self.emit(_make.IfThenElse(cond, stmt, None))
        return WithScope(None, _exit_cb)

    def _else(self):
        cb = Stage.get_cb()
        prev = cb.stmt_stack[-1][-1]
        cb.stmt_stack[-1].pop()
        cb.stmt_stack.append([])
        def _exit_cb():
            stmt = self.pop_stmt()
            self.has_break = False
            self.emit(self.replace_else(prev, stmt))
        return WithScope(None, _exit_cb)

    def _elif(self, cond):
        cb = Stage.get_cb()
        prev = cb.stmt_stack[-1][-1]
        cb.stmt_stack[-1].pop()
        cb.stmt_stack.append([])
        def _exit_cb():
            stmt = self.pop_stmt()
            self.has_break = False
            self.emit(self.replace_else(prev, _make.IfThenElse(cond, stmt, None)))
        return WithScope(None, _exit_cb)

    def _for(self, begin, end, step=1, name=None, dtype="int32", for_type="serial"):
        cb = Stage.get_cb()
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
        cb = Stage.get_cb()
        cb.stmt_stack.append([])
        self.for_level += 1
        def _exit_cb():
            stmt = self.pop_stmt()
            self.has_break = False
            self.for_level -= 1
            self.emit(_make.While(cond, stmt))
        return WithScope(None, _exit_cb)

    def _for_itervar(self, var, for_type_id = 0):
        cb = Stage.get_cb()
        cb.stmt_stack.append([])
        cb.var_dict[var.var.name] = var
        cb.axis_list.append(var)
        def _exit_cb():
            if isinstance(var, (list, tuple)):
                self.emit(util.make_for(var, self.pop_stmt(), 0))
            else:
                stmt = _make.AttrStmt(var, "loop_scope", var.var, self.pop_stmt())
                self.emit(_make.For(var.var, var.dom.min, var.dom.extent, for_type_id, 0, stmt))
        return WithScope(None, _exit_cb)
