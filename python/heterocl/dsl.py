from .tvm import make as _make
from .tvm import ir_pass as _pass
from .tvm.api import _IterVar
from .tvm.ir_builder import WithScope
from .schedule import Stage
from . import util

def if_(cond):
    assert Stage.get_len() > 0, "Incorrect usage of if_"
    stage = Stage.get_current()
    stage.stmt_stack.append([])
    def _exit_cb():
        stmt = stage.pop_stmt()
        stage.has_break = False
        stage.emit(_make.IfThenElse(cond, stmt, None))
    return WithScope(None, _exit_cb)

def else_():
    assert Stage.get_len() > 0, "Incorrect usage of else_"
    stage = Stage.get_current()
    prev = stage.stmt_stack[-1][-1]
    stage.stmt_stack[-1].pop()
    stage.stmt_stack.append([])
    def _exit_cb():
        stmt = stage.pop_stmt()
        stage.has_break = False
        stage.emit(stage.replace_else(prev, stmt))
    return WithScope(None, _exit_cb)

def elif_(cond):
    assert Stage.get_len() > 0, "Incorrect usage of elif_"
    stage = Stage.get_current()
    prev = stage.stmt_stack[-1][-1]
    stage.stmt_stack[-1].pop()
    stage.stmt_stack.append([])
    def _exit_cb():
        stmt = stage.pop_stmt()
        stage.has_break = False
        stage.emit(stage.replace_else(prev, _make.IfThenElse(cond, stmt, None)))
    return WithScope(None, _exit_cb)

def for_(begin, end, step=1, name="i", dtype="int32", for_type="serial"):
    assert Stage.get_len() > 0, "Incorrect usage of for_"
    stage = Stage.get_current()
    stage.stmt_stack.append([])
    extent = (end - begin)/step
    extent = util.CastRemover().mutate(extent)
    name = "i"+str(cb.for_ID) if name is None else name
    stage.for_ID += 1
    iter_var = _IterVar((0, extent), name, 0)
    stage.var_dict[name] = iter_var
    stage.axis_list.append(iter_var)
    stage.for_level += 1
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
        stmt = _make.AttrStmt(iter_var, "loop_scope", iter_var.var, stage.pop_stmt())
        stage.has_break = False
        stage.for_level -= 1
        stage.emit(_make.For(iter_var.var, 0, extent, for_type_id, 0, stmt))
    ret_var = _pass.Simplify(iter_var.var * step + begin)
    return WithScope(ret_var, _exit_cb)

def while_(cond):
    assert Stage.get_len() > 0, "Incorrect usage of while_"
    stage = Stage.get_current()
    stage.stmt_stack.append([])
    stage.for_level += 1
    def _exit_cb():
        stmt = stage.pop_stmt()
        stage.has_break = False
        stage.for_level -= 1
        stage.emit(_make.While(cond, stmt))
    return WithScope(None, _exit_cb)

def or_(*args):
    ret = args[0]
    for i in range(1, len(args)):
        ret = _make.Or(ret, args[i])
    return ret

def and_(*args):
    ret = args[0]
    for i in range(1, len(args)):
        ret = _make.And(ret, args[i])
    return ret

def break_():
    assert Stage.get_len() > 0, "Incorrect usage of break_"
    assert Stage.get_current().for_level > 0, "Break must be used inside a for/while loop"
    Stage.get_current().emit(_make.Break())
    Stage.get_current().has_break = True

def return_(val):
    assert Stage.get_len() > 0, "Incorrect usage of return_"
    stage = Stage.get_current()
    dtype = util.get_dtype(stage.ret_dtype)
    stage.emit(_make.Return(_make.Cast(dtype, val)))
    stage.has_return = True
