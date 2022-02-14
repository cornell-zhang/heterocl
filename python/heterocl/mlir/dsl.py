import hcl_mlir
from hcl_mlir.dialects import affine

from .context import ImperativeLoopDepth, ImperativeLoopNestCount, StageName
from .schedule import Schedule


class WithScope(object):
    """Auxiliary scope with"""

    def __init__(self, enter_value, exit_cb):
        self._enter_value = enter_value
        self._exit_cb = exit_cb

    def __enter__(self):
        return self._enter_value

    def __exit__(self, ptype, value, trace):
        self._exit_cb()


def for_(begin, end, step=1, name="i"):
    """Construct a FOR loop.

    Be careful: should not be used with other compute APIs like sum
    """
    depth = ImperativeLoopDepth.get()
    count = ImperativeLoopNestCount.get()
    stage = StageName.get() if depth == 0 else ""
    if depth == 0:
        ImperativeLoopNestCount.set(count + 1)
    ImperativeLoopDepth.set(depth + 1)
    hcl_mlir.enable_build_inplace()
    # TODO(Niansong): loop bounds must be expressions of itervar, e.g. k+1
    if isinstance(begin, (int, hcl_mlir.IterVar)) and isinstance(end, (int, hcl_mlir.IterVar)):
        loop = hcl_mlir.make_affine_for(
            begin, end, step, name=name, stage=stage, ip=hcl_mlir.GlobalInsertionPoint.get())
    else:
        raise RuntimeError("Not implemented")
    iter_var = hcl_mlir.IterVar(loop.induction_variable)
    hcl_mlir.GlobalInsertionPoint.save(loop.body)

    def _exit_cb():
        affine.AffineYieldOp([], ip=hcl_mlir.GlobalInsertionPoint.get())
        hcl_mlir.GlobalInsertionPoint.restore()
        ImperativeLoopDepth.set(ImperativeLoopDepth.get() - 1)

    return WithScope(iter_var, _exit_cb)


def if_(cond):
    """Construct an IF branch.
    """
    hcl_mlir.enable_build_inplace()
    if isinstance(cond, hcl_mlir.ExprOp):
        if_op = hcl_mlir.make_if(cond, ip=hcl_mlir.GlobalInsertionPoint.get())
    else:
        raise RuntimeError("Not implemented")
    hcl_mlir.GlobalInsertionPoint.save(if_op.then_block)
    Schedule._IfElseStack.append(if_op)

    def _exit_cb():
        hcl_mlir.GlobalInsertionPoint.restore()

    return WithScope(None, _exit_cb)


def else_():
    """Construct an ELSE branch.
    """
    hcl_mlir.enable_build_inplace()
    if len(Schedule._IfElseStack) == 0:
        raise RuntimeError("There is no if_ in front of the else_ branch")
    last_if_op = Schedule._IfElseStack.pop()
    hcl_mlir.GlobalInsertionPoint.save(last_if_op.else_block)

    def _exit_cb():
        hcl_mlir.GlobalInsertionPoint.restore()

    return WithScope(None, _exit_cb)


def elif_(cond):
    """Construct an ELIF branch.
    """
    # TODO: cond's built location is incorrect
    hcl_mlir.enable_build_inplace()
    if len(Schedule._IfElseStack) == 0:
        raise RuntimeError(
            "There is no if_ or elif_ in front of the elif_ branch")
    last_if_op = Schedule._IfElseStack.pop()
    hcl_mlir.GlobalInsertionPoint.save(last_if_op.else_block)

    if isinstance(cond, hcl_mlir.ExprOp):
        if_op = hcl_mlir.make_if(cond, ip=hcl_mlir.GlobalInsertionPoint.get())
    else:
        raise RuntimeError("Not implemented")
    hcl_mlir.GlobalInsertionPoint.save(if_op.then_block)
    Schedule._IfElseStack.append(if_op)

    def _exit_cb():
        hcl_mlir.GlobalInsertionPoint.restore()
        hcl_mlir.GlobalInsertionPoint.restore()

    return WithScope(None, _exit_cb)
