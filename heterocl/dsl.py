# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument

from hcl_mlir.exceptions import HCLDeprecationWarning

from .context import UniqueName
from .schedule import Schedule
from .utils import get_src_loc
from .ast import ast
from .types import UInt


class WithScope:
    """Auxiliary scope with"""

    def __init__(self, enter_value, exit_cb):
        self._enter_value = enter_value
        self._exit_cb = exit_cb

    def __enter__(self):
        return self._enter_value

    def __exit__(self, ptype, value, trace):
        self._exit_cb()


def and_(*args):
    """Compute the logic AND between expressions."""
    filename, loc = get_src_loc()
    loc = ast.Location(filename, loc)
    # pylint: disable=redefined-variable-type
    expr = ast.ConstantOp(1, UInt(1), loc)
    for arg in args:
        arg = ast.CastOp(arg, UInt(1), loc)
        expr = ast.LogicalAnd(expr, arg, loc)
    return expr


def or_(*args):
    """Compute the logic OR between expressions."""
    filename, loc = get_src_loc()
    loc = ast.Location(filename, loc)
    # pylint: disable=redefined-variable-type
    expr = ast.ConstantOp(0, UInt(1), loc)
    for arg in args:
        arg = ast.CastOp(arg, UInt(1), loc)
        expr = ast.LogicalOr(expr, arg, loc)
    return expr


def not_(arg):
    """Compute the logic NOT operation."""
    filename, loc = get_src_loc()
    loc = ast.Location(filename, loc)
    one = ast.ConstantOp(1, UInt(1), loc)
    arg = ast.CastOp(arg, UInt(1), loc)
    return ast.LogicalXOr(arg, one, loc)


def for_(begin, end, step=1, tag=None, name=None):
    """Construct a FOR loop.

    Be careful: should not be used with other compute APIs like sum
    """

    name = UniqueName.get(name, "loop")

    region = ast.scope.get()
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    forOp = ast.ForOp(tag, name, begin, end, step, loc)
    region.append(forOp)
    ast.scope.push(forOp.body)

    def _exit_cb():
        ast.scope.pop()

    return WithScope(forOp.iter_var, _exit_cb)


def if_(cond):
    region = ast.scope.get()
    filename, lineno = get_src_loc()
    ifOp = ast.IfOp(cond, ast.Location(filename, lineno))
    region.append(ifOp)
    ast.scope.push(ifOp.body)

    def _exit_cb():
        ast.scope.pop()

    return WithScope(None, _exit_cb)


def else_():
    region = ast.scope.get()
    filename, lineno = get_src_loc()
    elseOp = ast.ElseOp(ast.Location(filename, lineno))
    region.append(elseOp)
    ast.scope.push(elseOp.body)

    def _exit_cb():
        ast.scope.pop()

    return WithScope(None, _exit_cb)


def elif_(cond):
    region = ast.scope.get()
    filename, lineno = get_src_loc()
    elifOp = ast.ElseIfOp(cond, ast.Location(filename, lineno))
    region.append(elifOp)
    ast.scope.push(elifOp.body)

    def _exit_cb():
        ast.scope.pop()

    return WithScope(None, _exit_cb)


def while_(cond):
    region = ast.scope.get()
    filename, lineno = get_src_loc()
    whileOp = ast.WhileOp(cond, ast.Location(filename, lineno))
    region.append(whileOp)
    ast.scope.push(whileOp.body)

    def _exit_cb():
        ast.scope.pop()

    return WithScope(None, _exit_cb)


def def_(shapes=None, dtypes=None, ret_dtype=None, name=None, arg_names=None):
    """
    issue warning if any arg is not None
    assumption: return is the terminator of func region
    """
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)

    def decorator(fmodule):
        HCLDeprecationWarning(
            "hcl.def_() is deprecated, please use .outline() instead."
        ).warn()
        fname = fmodule.__name__
        region = ast.scope.get()
        func_op = ast.FuncOp(fname, [], [], loc)
        region.append(func_op)

        def wrapped_func(*inputs):
            # build a function signature
            func_sig = fname + "("
            arg_shapes = [v.shape for v in inputs]
            func_sig += ", ".join([str(s) for s in arg_shapes])
            func_sig += ")"

            if func_sig not in Schedule._FuncDefs:
                func_op.args = inputs
                ast.scope.push(func_op.body)
                ret = fmodule(*inputs)
                ast.scope.pop()
                if ret is None:
                    outputs = []
                    if len(func_op.body) > 0 and isinstance(
                        func_op.body[-1], ast.ReturnOp
                    ):
                        outputs = [func_op.body[-1].expr]
                        func_op.body.pop()
                elif isinstance(ret, tuple):
                    outputs = list(ret)
                else:
                    outputs = [ret]

                func_op.return_tensors.extend(outputs)
                Schedule._FuncDefs[func_sig] = func_op

            else:
                outputs = Schedule._FuncDefs[func_sig].return_tensors

            # call op
            call_op = ast.CallOp(func_op.name, inputs, outputs, loc)
            if len(outputs) == 0:
                # no return value
                # use as a statement
                region = ast.scope.get()
                region.append(call_op)
            else:
                # return value
                # use as an expression
                call_op.level = 0

            return call_op

        return wrapped_func

    return decorator


def return_(expr=None):
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    return_op = ast.ReturnOp(expr, loc)
    region = ast.scope.get()
    region.append(return_op)


def break_():
    raise RuntimeError(
        "Currently we cannot support hcl.break_ due to MLIR's limitation. Please rewrite your prorgam."
    )
