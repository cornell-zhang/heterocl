import hcl_mlir
from hcl_mlir import GlobalInsertionPoint
from hcl_mlir.dialects import affine, builtin
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.dialects import scf, std
from hcl_mlir.ir import *

from .. import config
from .context import (ImperativeLoopDepth, ImperativeLoopNestCount,
                      NestedCompute, StageName, UniqueName)
from .schedule import Schedule, Stage
from .utils import get_extra_type_hints, hcl_dtype_to_mlir


class WithScope(object):
    """Auxiliary scope with"""

    def __init__(self, enter_value, exit_cb):
        self._enter_value = enter_value
        self._exit_cb = exit_cb

    def __enter__(self):
        return self._enter_value

    def __exit__(self, ptype, value, trace):
        self._exit_cb()


def any(*args):
    """Create a new experssion of the union of all conditions in the arguments
    """
    if not args:
        raise ValueError("Any must take at least 1 argument")
    if len(args) == 1:
        return args[0]
    ret = hcl_mlir.OrOp(args[0], args[1])
    for i in range(2, len(args)):
        ret = hcl_mlir.OrOp(ret, args[i])
    return ret


def all(*args):
    """Create a new experssion of the intersection of all conditions in the
      arguments
    """
    if not args:
        raise ValueError("Any must take at least 1 argument")
    if len(args) == 1:
        return args[0]
    ret = hcl_mlir.AndOp(args[0], args[1])
    for i in range(2, len(args)):
        ret = hcl_mlir.AndOp(ret, args[i])
    return ret


def and_(*args):
    """Compute the logic AND between expressions.
    """
    return all(*args)


def or_(*args):
    """Compute the logic OR between expressions.
    """
    return any(*args)


def for_(begin, end, step=1, tag=""):
    """Construct a FOR loop.

    Be careful: should not be used with other compute APIs like sum
    """
    depth = ImperativeLoopDepth.get()
    count = ImperativeLoopNestCount.get()
    if tag == None:
        stage_name = StageName.get()
    else:
        stage_name = tag
    if depth == 0:
        Schedule._CurrentStage.append(Stage(stage_name))
        Schedule._TopFunction.__setattr__(
            stage_name, Schedule._CurrentStage[-1])
        ImperativeLoopNestCount.set(count + 1)
    ImperativeLoopDepth.set(depth + 1)

    hcl_mlir.enable_build_inplace()
    # TODO(Niansong): loop bounds must be expressions of itervar, e.g. k+1
    if isinstance(begin, (int, hcl_mlir.IterVar)) and isinstance(end, (int, hcl_mlir.IterVar)):
        loop_name = UniqueName.get("loop")
        loop_handle = hcl_d.CreateLoopHandleOp(StringAttr.get(
            loop_name), ip=hcl_mlir.GlobalInsertionPoint.ip_stack[-depth-1])
        loop = hcl_mlir.make_affine_for(
            begin, end, step, name=loop_name, stage=stage_name, ip=hcl_mlir.GlobalInsertionPoint.get())
        Schedule._CurrentStage[-1].add_axis(loop_handle)
    else:
        raise RuntimeError("Not implemented")
    iter_var = hcl_mlir.IterVar(loop.induction_variable, name=stage_name)
    if step < 0:
        hcl_mlir.GlobalInsertionPoint.save(loop.body)
        begin = hcl_mlir.ConstantOp("index", begin)
        iter_var = begin - iter_var
        hcl_mlir.GlobalInsertionPoint.restore()
    hcl_mlir.GlobalInsertionPoint.save(loop.body)

    def _exit_cb():
        if isinstance(loop, affine.AffineForOp):
            affine.AffineYieldOp([], ip=hcl_mlir.GlobalInsertionPoint.get())
        else:
            scf.YieldOp([], ip=hcl_mlir.GlobalInsertionPoint.get())
        hcl_mlir.GlobalInsertionPoint.restore()
        if depth == 0:
            # itself
            Schedule._CurrentStage[-1].set_output(Schedule._CurrentStage[-1])
            Schedule._CurrentStage[-1].done()
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
        if isinstance(if_op, affine.AffineIfOp):
            affine.AffineYieldOp([], ip=hcl_mlir.GlobalInsertionPoint.get())
        else:
            scf.YieldOp([], ip=hcl_mlir.GlobalInsertionPoint.get())
        hcl_mlir.GlobalInsertionPoint.restore()

    return WithScope(None, _exit_cb)


def else_():
    """Construct an ELSE branch.
    """
    hcl_mlir.enable_build_inplace()
    if len(Schedule._IfElseStack) == 0:
        raise RuntimeError("There is no if_ in front of the else_ branch")
    last_if_op = Schedule._IfElseStack.pop()
    last_if_op.regions[1].blocks.append(*[])
    hcl_mlir.GlobalInsertionPoint.save(last_if_op.else_block)

    def _exit_cb():
        if isinstance(last_if_op, affine.AffineIfOp):
            affine.AffineYieldOp([], ip=hcl_mlir.GlobalInsertionPoint.get())
        else:
            scf.YieldOp([], ip=hcl_mlir.GlobalInsertionPoint.get())
        hcl_mlir.GlobalInsertionPoint.restore()

    return WithScope(None, _exit_cb)


def elif_(cond):
    """Construct an ELIF branch.
    """
    hcl_mlir.enable_build_inplace()
    if len(Schedule._IfElseStack) == 0:
        raise RuntimeError(
            "There is no if_ or elif_ in front of the elif_ branch")
    last_if_op = Schedule._IfElseStack.pop()
    last_if_op.regions[1].blocks.append(*[])
    hcl_mlir.GlobalInsertionPoint.save(last_if_op.else_block)

    if isinstance(cond, hcl_mlir.ExprOp):
        if_op = hcl_mlir.make_if(cond, ip=hcl_mlir.GlobalInsertionPoint.get())
    else:
        raise RuntimeError("Not implemented")
    hcl_mlir.GlobalInsertionPoint.save(if_op.then_block)
    Schedule._IfElseStack.append(if_op)

    def _exit_cb():
        if isinstance(if_op, affine.AffineIfOp):
            affine.AffineYieldOp([], ip=hcl_mlir.GlobalInsertionPoint.get())
        else:
            scf.YieldOp([], ip=hcl_mlir.GlobalInsertionPoint.get())
        hcl_mlir.GlobalInsertionPoint.restore()
        if isinstance(last_if_op, affine.AffineIfOp):
            affine.AffineYieldOp([], ip=hcl_mlir.GlobalInsertionPoint.get())
        else:
            scf.YieldOp([], ip=hcl_mlir.GlobalInsertionPoint.get())
        hcl_mlir.GlobalInsertionPoint.restore()

    return WithScope(None, _exit_cb)


def while_(cond):
    """Construct an IF branch.
    """
    hcl_mlir.enable_build_inplace()
    if isinstance(cond, hcl_mlir.ExprOp):
        while_op = hcl_mlir.make_while(
            cond, ip=hcl_mlir.GlobalInsertionPoint.get())
    else:
        raise RuntimeError("Not implemented")
    hcl_mlir.GlobalInsertionPoint.save(while_op.after.blocks[0])

    def _exit_cb():
        scf.YieldOp([], ip=hcl_mlir.GlobalInsertionPoint.get())
        hcl_mlir.GlobalInsertionPoint.restore()

    return WithScope(None, _exit_cb)


DEF_FUNC = False


def def_(shapes, dtypes=None, ret_dtype=None, name=None, arg_names=None):
    """
    Define a HeteroCL function from a Python function.

    Actual execution order:
    (def_(shapes))(fmodule(*args))
    """

    def decorator(fmodule):
        fname = name if name is not None else fmodule.__name__
        if Schedule._TopFunction != None:
            Schedule._TopFunction.__setattr__(fname, fmodule)
        code = fmodule.__code__
        names = code.co_varnames
        if arg_names is not None:
            names = list(names)
            for i in range(len(arg_names)):
                names[i] = arg_names[i]
            names = tuple(names)
        nargs = code.co_argcount

        # prepare input types
        input_types = []
        input_elt = []
        if dtypes is None:
            dtype = config.init_dtype
            dtype = hcl_dtype_to_mlir(dtype)
            if hcl_mlir.is_unsigned_type(dtype):
                dtype = IntegerType.get_signless(dtype.width)
            for shape in shapes:
                input_elt.append(dtype)
                if shape != ():  # scalar
                    input_types.append(MemRefType.get(shape, dtype))
                else:
                    input_types.append(dtype)
        elif isinstance(dtypes, list):
            if len(dtypes) != nargs:
                raise RuntimeError(
                    "The number of data types does not match the of arguments")
            for shape, dtype in zip(shapes, dtypes):
                dtype = hcl_dtype_to_mlir(dtype)
                if hcl_mlir.is_unsigned_type(dtype):
                    dtype = IntegerType.get_signless(dtype.width)
                input_elt.append(dtype)
                if shape != ():  # scalar
                    input_types.append(MemRefType.get(shape, dtype))
                else:
                    input_types.append(dtype)
        else:
            raise RuntimeError("Unrecognized dtype format")
        # prepare return types
        return_types = []
        return_elt = []
        if ret_dtype is not None:
            dtype = hcl_dtype_to_mlir(ret_dtype)
            return_elt.append(dtype)
            return_types.append(dtype)

        # create stage function
        stage_func_name = "Stage_" + fname
        # here we also put the return in the input argument,
        # since commonly in C++ we should pass the array by reference
        stage_func_op = builtin.FuncOp(name=stage_func_name, type=FunctionType.get(
            inputs=input_types+return_types, results=[]), ip=GlobalInsertionPoint.ip_stack[0])
        # stage_func_op.attributes["inputs"] = StringAttr.get(
        #     ",".join([tensor.name for tensor in self.inputs]))
        stage_func_op.attributes["extra_itypes"] = StringAttr.get("".join([get_extra_type_hints(
            dtype) for dtype in input_elt] + [get_extra_type_hints(dtype) for dtype in return_elt]))  # inputs & outputs
        # if self.output is not None:
        #     stage_func_op.attributes["outputs"] = StringAttr.get(
        #         self.output.op.name)
        stage_func_op.add_entry_block()

        def wrapped_func(*inputs):
            global DEF_FUNC
            DEF_FUNC = True
            hcl_mlir.enable_build_inplace()
            # call this function in the top function
            call_arglist = []
            for i, tensor in enumerate(inputs):
                call_arglist.append(tensor.result)
                if isinstance(tensor, hcl_mlir.IterVar):
                    input_types[i] = IndexType.get()
                    stage_func_op.entry_block.arguments[i].set_type(
                        IndexType.get())
            # update function type
            stage_func_op.attributes["type"] = TypeAttr.get(
                FunctionType.get(inputs=input_types+return_types, results=[]))

            # update inner load/store references
            # used for recovery
            original_tensor_op = []
            for tensor, arg in zip(inputs, stage_func_op.entry_block.arguments):
                if isinstance(tensor, hcl_mlir.IterVar):
                    original_tensor_op.append(tensor.built_op)
                    tensor.op = arg
                    tensor.built_op = arg
                elif isinstance(tensor.op, hcl_mlir.TensorOp):
                    original_tensor_op.append(tensor.op.built_op)
                    tensor.op.update_op(arg)
                else:  # ComputeOp
                    original_tensor_op.append(tensor.op.output.op.built_op)
                    tensor.op.output.op.update_op(arg)
            # insertion point become the stage function inside
            GlobalInsertionPoint.save(
                InsertionPoint(stage_func_op.entry_block))

            # execute the original function
            fmodule(*inputs)

            # recover as top function op
            for i, tensor in enumerate(inputs):
                if isinstance(tensor, hcl_mlir.IterVar):
                    tensor.op = original_tensor_op[i]
                    tensor.built_op = original_tensor_op[i]
                elif isinstance(tensor.op, hcl_mlir.TensorOp):
                    tensor.op.update_op(original_tensor_op[i])
                else:  # ComputeOp
                    tensor.op.output.op.update_op(
                        original_tensor_op[i])

            # recover from the subfunction
            if len(Schedule._DefFuncReturn) == 0:
                ret_op = std.ReturnOp([], ip=GlobalInsertionPoint.get())
                GlobalInsertionPoint.restore()
                # build call op
                call_op = hcl_mlir.CallOp(None, stage_func_name, call_arglist)
                call_op.built_op.attributes["inputs"] = StringAttr.get(
                    ",".join([tensor.name for tensor in inputs]))
            else:
                if Schedule._DefFuncReturn[0] is not None:
                    new_return_types = [Schedule._DefFuncReturn[0].dtype]
                else:
                    new_return_types = []
                stage_func_op.attributes["type"] = TypeAttr.get(
                    FunctionType.get(inputs=input_types, results=new_return_types))
                GlobalInsertionPoint.restore()
                # build call op
                call_op = hcl_mlir.CallOp(
                    Schedule._DefFuncReturn[0].dtype, stage_func_name, call_arglist)
                call_op.built_op.attributes["inputs"] = StringAttr.get(
                    ",".join([tensor.name for tensor in inputs]))
                # call_op.built_op.attributes["outputs"] = StringAttr.get(
                #     Schedule._DefFuncReturn[0].name)

            DEF_FUNC = False
            return call_op

        return wrapped_func

    return decorator


def return_(expr=None):
    hcl_mlir.enable_build_inplace()
    if expr is not None:
        if DEF_FUNC:  # imperative
            expr = hcl_mlir.get_hcl_op(expr)
            Schedule._DefFuncReturn.append(expr)
            ret_op = std.ReturnOp(
                [expr.result], ip=hcl_mlir.GlobalInsertionPoint.get())
            hcl_mlir.GlobalInsertionPoint.ip_stack[-1] = InsertionPoint(ret_op)
        elif isinstance(expr, (int, float, hcl_mlir.IterVar)) or expr.built_op == None:  # declarative
            expr = hcl_mlir.get_hcl_op(expr)
            builder = hcl_mlir.ASTVisitor("build")
            builder.visit(expr)
            hcl_mlir.StoreOp(expr, Schedule._CurrentStage[-1].op.op,
                             Schedule._CurrentStage[-1].op.iter_var)
            ret_op = Schedule._CurrentStage[-1].op
        else:
            raise RuntimeError("Not recognized return value")
    else:
        Schedule._DefFuncReturn.append(None)
        ret_op = std.ReturnOp(
            [], ip=hcl_mlir.GlobalInsertionPoint.get())
    return ret_op
