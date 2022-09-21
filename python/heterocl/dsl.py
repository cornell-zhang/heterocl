import inspect

import hcl_mlir
from hcl_mlir import GlobalInsertionPoint
from hcl_mlir.dialects import affine
from hcl_mlir.dialects import func as func_d
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.dialects import scf
from hcl_mlir.ir import *
from hcl_mlir.exceptions import *

from . import config
from .context import (BreakFlag, ImperativeLoopDepth, ImperativeLoopNestCount,
                      NestedStageLevel, StageName, UniqueName, IPPointer)
from .schedule import Schedule, Stage
from .tensor import Tensor
from .utils import get_extra_type_hints, hcl_dtype_to_mlir, get_func_obj


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
    a | b = !(!a & !b)
    """
    if not args:
        raise ValueError("Any must take at least 1 argument")
    ret = hcl_mlir.LogicalOrOp(*args)
    return ret


def all(*args):
    """Create a new experssion of the intersection of all conditions in the
    arguments
    """
    if not args:
        raise ValueError("Any must take at least 1 argument")
    ret = hcl_mlir.LogicalAndOp(*args)
    return ret


def and_(*args):
    """Compute the logic AND between expressions."""
    return all(*args)


def or_(*args):
    """Compute the logic OR between expressions."""
    return any(*args)


def not_(arg):
    """Compute the logic NOT operation.
    """
    return hcl_mlir.LogicalNotOp(arg)


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
        IPPointer.set(len(hcl_mlir.GlobalInsertionPoint.ip_stack) - 1)
        Schedule._CurrentStage.append(Stage(stage_name))
        Schedule._CurrentStage[-1].stage_handle = hcl_d.CreateOpHandleOp(
            StringAttr.get(stage_name), ip=hcl_mlir.GlobalInsertionPoint.ip_stack[IPPointer.get()]
        )
        Schedule._TopFunction.__setattr__(
            stage_name, Schedule._CurrentStage[-1])
        ImperativeLoopNestCount.set(count + 1)
    ImperativeLoopDepth.set(depth + 1)

    hcl_mlir.enable_build_inplace()
    loop_name = UniqueName.get("loop")
    loop_nest_ip = IPPointer.get()
    loop_handle = hcl_d.CreateLoopHandleOp(Schedule._CurrentStage[-1].stage_handle.result, StringAttr.get(
        loop_name), ip=hcl_mlir.GlobalInsertionPoint.ip_stack[loop_nest_ip])
    loop = hcl_mlir.make_for(
        begin, end, step, name=loop_name, stage=stage_name, ip=hcl_mlir.GlobalInsertionPoint.get())
    Schedule._CurrentLoops.append(loop)
    Schedule._CurrentStage[-1].add_axis(loop_handle)

    iter_var = hcl_mlir.IterVar(loop.induction_variable, name=stage_name)
    hcl_mlir.GlobalInsertionPoint.save(loop.body.operations[0])
    if step < 0:
        begin = hcl_mlir.ConstantOp("index", begin)
        iter_var = begin - iter_var

    def _exit_cb():
        if BreakFlag.get():
            hcl_mlir.GlobalInsertionPoint.restore()
            BreakFlag.set(False)
        hcl_mlir.GlobalInsertionPoint.restore()
        ImperativeLoopDepth.set(ImperativeLoopDepth.get() - 1)
        if depth == 0:
            # Setting the stage output, i.e. .op attribute as the stage itself
            Schedule._CurrentStage[-1].set_output(Schedule._CurrentStage[-1])
            Schedule._CurrentStage[-1].done()
            # attach this stage to the caller function
            if Schedule._TopFunction is None:
                raise APIError("No top function is found. Imperative function must be called by a top-level function.")
            caller_func_name = inspect.stack()[1].function
            called_from_top = caller_func_name == Schedule._TopFunction.__name__
            caller_func = get_func_obj(caller_func_name)
            if not called_from_top:
                # Caller function up one level
                caller_parent_func_name = inspect.stack()[2].function
                caller_parent_func = get_func_obj(caller_parent_func_name)
                # If haven't already, attach the caller function
                # to its parent function as an attribute
                if not hasattr(caller_parent_func, caller_func_name):
                    caller_parent_func.__setattr__(caller_func_name, caller_func)
            stage = Schedule._CurrentStage[-1]
            caller_func.__setattr__(stage.name, stage)
            # Set up a list of stages for the caller function
            if not hasattr(caller_func, "_stages"):
                caller_func.__setattr__("_stages", [stage])
            else:
                caller_func._stages.append(stage)    

    return WithScope(iter_var, _exit_cb)


def if_(cond):
    """Construct an IF branch."""
    hcl_mlir.enable_build_inplace()
    if isinstance(cond, hcl_mlir.ExprOp):
        if_op = hcl_mlir.make_if(cond, ip=hcl_mlir.GlobalInsertionPoint.get())
    else:
        raise RuntimeError("Not implemented")
    hcl_mlir.GlobalInsertionPoint.save(if_op.then_block.operations[0])
    Schedule._IfElseStack.append(if_op)

    def _exit_cb():
        if BreakFlag.get():
            ip_stack = hcl_mlir.GlobalInsertionPoint.ip_stack
            hcl_mlir.GlobalInsertionPoint.ip_stack = ip_stack[:-2] + [
                ip_stack[-1]]
        else:
            hcl_mlir.GlobalInsertionPoint.restore()

    return WithScope(None, _exit_cb)


def else_():
    """Construct an ELSE branch."""
    hcl_mlir.enable_build_inplace()
    if len(Schedule._IfElseStack) == 0:
        raise RuntimeError("There is no if_ in front of the else_ branch")
    last_if_op = Schedule._IfElseStack.pop()
    last_if_op.regions[1].blocks.append(*[])
    if isinstance(last_if_op, affine.AffineIfOp):
        affine.AffineYieldOp([], ip=InsertionPoint(last_if_op.else_block))
    else:
        scf.YieldOp([], ip=hcl_mlir.InsertionPoint(last_if_op.else_block))
    hcl_mlir.GlobalInsertionPoint.save(last_if_op.else_block.operations[0])

    def _exit_cb():
        hcl_mlir.GlobalInsertionPoint.restore()

    return WithScope(None, _exit_cb)


def elif_(cond):
    """Construct an ELIF branch."""
    hcl_mlir.enable_build_inplace()
    if len(Schedule._IfElseStack) == 0:
        raise RuntimeError(
            "There is no if_ or elif_ in front of the elif_ branch")
    last_if_op = Schedule._IfElseStack.pop()
    last_if_op.regions[1].blocks.append(*[])
    if isinstance(last_if_op, affine.AffineIfOp):
        affine.AffineYieldOp([], ip=InsertionPoint(last_if_op.else_block))
    else:
        scf.YieldOp([], ip=hcl_mlir.InsertionPoint(last_if_op.else_block))
    hcl_mlir.GlobalInsertionPoint.save(last_if_op.else_block.operations[0])

    if isinstance(cond, hcl_mlir.ExprOp):
        if_op = hcl_mlir.make_if(cond, ip=hcl_mlir.GlobalInsertionPoint.get())
    else:
        raise RuntimeError("Not implemented")
    hcl_mlir.GlobalInsertionPoint.save(if_op.then_block.operations[0])
    Schedule._IfElseStack.append(if_op)

    def _exit_cb():
        hcl_mlir.GlobalInsertionPoint.restore()
        hcl_mlir.GlobalInsertionPoint.restore()

    return WithScope(None, _exit_cb)


def while_(cond):
    """Construct a while loop"""
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
        warnings.warn(
            "hcl.def_() is deprecated, please use .outline() instead.", DeprecationWarning)
        if Schedule._CurrentSchedule is None:
            raise RuntimeError(
                "def_() must be called with hcl.create_schedule")
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
                    "The number of data types does not match the of arguments"
                )
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
        stage_func_op = func_d.FuncOp(
            name=stage_func_name,
            type=FunctionType.get(inputs=input_types +
                                  return_types, results=[]),
            ip=GlobalInsertionPoint.ip_stack[0],
        )
        # stage_func_op.attributes["inputs"] = StringAttr.get(
        #     ",".join([tensor.name for tensor in self.inputs]))
        stage_func_op.attributes["itypes"] = StringAttr.get(
            "".join(
                [get_extra_type_hints(dtype) for dtype in input_elt]
                + [get_extra_type_hints(dtype) for dtype in return_elt]
            )
        )  # inputs & outputs
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
            stage_func_op.attributes["function_type"] = TypeAttr.get(
                FunctionType.get(inputs=input_types + return_types, results=[])
            )

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
                    tensor.op.output.op.update_op(original_tensor_op[i])

            # recover from the subfunction
            if len(Schedule._DefFuncReturn) == 0:
                ret_op = func_d.ReturnOp([], ip=GlobalInsertionPoint.get())
                GlobalInsertionPoint.restore()
                # build call op
                call_op = hcl_mlir.CallOp(None, stage_func_name, call_arglist)
                call_op.built_op.attributes["inputs"] = StringAttr.get(
                    ",".join([tensor.name for tensor in inputs])
                )
            else:
                if Schedule._DefFuncReturn[0] is not None:
                    new_return_types = [Schedule._DefFuncReturn[0].dtype]
                else:
                    new_return_types = []
                stage_func_op.attributes["function_type"] = TypeAttr.get(
                    FunctionType.get(inputs=input_types,
                                     results=new_return_types)
                )
                GlobalInsertionPoint.restore()
                # build call op
                call_op = hcl_mlir.CallOp(
                    Schedule._DefFuncReturn[0].dtype, stage_func_name, call_arglist
                )
                call_op.built_op.attributes["inputs"] = StringAttr.get(
                    ",".join([tensor.name for tensor in inputs])
                )
                # call_op.built_op.attributes["outputs"] = StringAttr.get(
                #     Schedule._DefFuncReturn[0].name)

            DEF_FUNC = False
            return call_op

        return wrapped_func

    return decorator


def return_(expr=None):
    if len(Schedule._IfElseStack) > 0:
        raise RuntimeError(
            "hcl.return_ statement cannot be in a nested region due to MLIR's limitation. Please rewrite your program and use .outline() to create a new function.")
    hcl_mlir.enable_build_inplace()
    if expr is not None:
        if DEF_FUNC:  # imperative
            expr = hcl_mlir.get_hcl_op(expr)
            Schedule._DefFuncReturn.append(expr)
            ret_op = func_d.ReturnOp(
                [expr.result], ip=hcl_mlir.GlobalInsertionPoint.get())
            hcl_mlir.GlobalInsertionPoint.ip_stack[-1] = InsertionPoint(ret_op)
        elif (
            isinstance(expr, (int, float, hcl_mlir.ExprOp)
                       ) or expr.built_op == None
        ):  # declarative
            expr = hcl_mlir.get_hcl_op(expr)
            builder = hcl_mlir.ASTVisitor("build")
            builder.visit(expr)
            hcl_mlir.StoreOp(
                expr,
                Schedule._CurrentStage[-1].op.op,
                Schedule._CurrentStage[-1].op.iter_var,
            )
            ret_op = Schedule._CurrentStage[-1].op
        else:
            raise RuntimeError("Not recognized return value")
    else:
        Schedule._DefFuncReturn.append(None)
        ret_op = func_d.ReturnOp([], ip=hcl_mlir.GlobalInsertionPoint.get())
    return ret_op


def break_():
    raise RuntimeError(
        "Currently we cannot support hcl.break_ due to MLIR's limitation. Please rewrite your prorgam.")
    # hcl_mlir.enable_build_inplace()
    # BreakFlag.set(True)
    # if len(Schedule._IfElseStack) == 0:
    #     raise RuntimeError("There is no if_ before hcl.break_")
    # last_if_op = Schedule._IfElseStack.pop()
    # last_if_op.regions[1].blocks.append(*[])
    # if isinstance(last_if_op, affine.AffineIfOp):
    #     affine.AffineYieldOp([], ip=InsertionPoint(last_if_op.else_block))
    # else:
    #     scf.YieldOp([], ip=hcl_mlir.InsertionPoint(last_if_op.else_block))
    # hcl_mlir.GlobalInsertionPoint.save(last_if_op.else_block.operations[0])

# def break_():
#     BreakFlag.set(True)
#     hcl_mlir.enable_build_inplace()
#     bool = MemRefType.get((1,), IntegerType.get_signless(1))
#     # outside stage
#     global_ip = InsertionPoint(Schedule._CurrentSchedule.device_top.entry_block.operations[0])
#     flag = memref.AllocOp(bool, [], [], None, ip=global_ip) # inside top func
#     zero_idx = arith.ConstantOp(
#         IndexType.get(),
#         IntegerAttr.get(IndexType.get(), 0),
#         ip=global_ip,
#     )
#     true_value = arith.ConstantOp(
#         IntegerType.get_signless(1), IntegerAttr.get(IntegerType.get_signless(1), 1), ip=global_ip
#     )
#     store = affine.AffineStoreOp(
#         true_value.result,
#         flag.result,
#         [zero_idx.result],
#         ip=global_ip,
#     )
#     # inside the stage
#     false_value = arith.ConstantOp(
#         IntegerType.get_signless(1), IntegerAttr.get(IntegerType.get_signless(1), 0), ip=GlobalInsertionPoint.get()
#     )
#     store = affine.AffineStoreOp(
#         false_value.result,
#         flag.result,
#         [zero_idx.result],
#         ip=GlobalInsertionPoint.get(),
#     )
#     # at the beginning of the stage
#     stage_ip = InsertionPoint(Schedule._CurrentLoops[-1].body.operations[0])
#     load = affine.AffineLoadOp(
#         flag.result, [zero_idx.result], ip=stage_ip
#     ) # op0
#     if_op = scf.IfOp(load.result, ip=stage_ip, hasElse=False) # op1
#     yield_op = scf.YieldOp([], ip=InsertionPoint(if_op.then_block))
#     for i, op in enumerate(Schedule._CurrentLoops[-1].body.operations):
#         if i < 2 or isinstance(op, (affine.AffineYieldOp, scf.YieldOp)):
#             continue
#         op.move_before(yield_op)
#     if len(Schedule._IfElseStack) > 0:
#         GlobalInsertionPoint.ip_stack.insert(-1, InsertionPoint(yield_op))
#     # GlobalInsertionPoint.save(yield_op)
#     print(Schedule._CurrentSchedule.device_module)
