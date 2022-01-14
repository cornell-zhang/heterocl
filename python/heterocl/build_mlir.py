import io
from collections import OrderedDict
from ordered_set import OrderedSet
import inspect

import hcl_mlir
import hcl_mlir.affine as affine
from hcl_mlir import (ASTBuilder, get_context, get_insertion_point,
                      get_location, set_insertion_point)
from mlir import passmanager
from mlir.dialects import builtin, memref, std
from mlir.execution_engine import *
from mlir.ir import *

from .base import get_func_body, get_function, get_module
from .schedule import Stage, Schedule
from .tvm.schedule import create_schedule as tvm_create_schedule


def placeholder(shape, name=None, dtype=None):
    """Construct a HeteroCL placeholder for inputs/outputs.
    """
    with get_context() as ctx, get_location() as loc:
        memref_type = MemRefType.get(shape, F32Type.get(ctx), loc=loc)
        tensor = hcl_mlir.TensorOp(shape, memref.AllocOp, memref_type)
        return tensor


def create_schedule(inputs, func, name=""):
    """Create a schedule for compute optimizations.
    """
    outputs = []
    if not isinstance(inputs, list):
        inputs = [inputs]
    # reset the global variables
    Schedule.stage_ops = []
    Schedule.mod_calls = dict()
    Schedule.stage_names = set()
    Schedule.last_stages = OrderedSet([])
    # create exact HCL IR nodes
    with get_context() as ctx, get_location() as loc, Stage("_top") as top:
        # create exact memref alloc
        for tensor in inputs:
            tensor.op = tensor.op(
                tensor.memref_type, None, None, None, ip=get_insertion_point()
            )
        # execute fcompute and generate inner IR nodes
        ret = func(*inputs)

    # append the output tensors to the input list
    if ret is not None:
        if isinstance(ret, tuple):
            outputs = list(ret)
        else:
            outputs.append(ret)
    # let each stage be an attribute of the function
    for op in top.substages:
        func.__setattr__(op.name, op)
    t = Schedule.last_stages
    ops = [t_._op.op for t_ in t]
    s = Schedule(tvm_create_schedule(ops), inputs, outputs, name)
    return s


def build(schedule, target=None, name="default_function", stmt=None):
    """Build the executable according to the schedule and target.
    """
    new_inputs = []
    for input_tensor in schedule.inputs:  # should be hcl_mlir.TensorOp
        new_inputs.append(input_tensor)

    with get_context(), get_location():
        # add block terminator
        std.ReturnOp([], ip=get_insertion_point())

        # apply schedule and lower
        func = get_function()
        hcl_mlir.loop_transformation(func.operation)
        get_module().dump()

    if target == "vhls":
        return build_fpga_kernel(schedule, new_inputs, target, name, stmt)
    else:
        return build_llvm(schedule, new_inputs, target, name, stmt)


def compute_mlir(shape, fcompute, name=None, dtype=None, attrs=OrderedDict()):
    """Construct a new tensor based on the shape and the compute function.
    """
    # check API correctness
    if not isinstance(shape, tuple):
        raise RuntimeError("The shape of compute API must be a tuple")

    shape = tuple([int(s) if isinstance(s, float) else s for s in shape])
    out_ndim = len(shape)

    argspec = inspect.getfullargspec(fcompute)
    if len(argspec.args) == 0 and argspec.varargs is None:
        arg_names = ["i%d" % i for i in range(out_ndim)]
    elif argspec.varargs is not None:
        # if there is a varargs, it takes the remaining dimensions of out_ndim
        arg_names = argspec.args + [
            f"i{i}" for i in range(out_ndim - len(argspec.args))
        ]
    else:
        arg_names = argspec.args
        # if there are fewer args than out dimensions, the remaining dimensions
        # are implicitly broadcast
        out_ndim = len(arg_names)
    assert argspec.varkw is None, "Variable keyword arguments not supported in fcompute"
    assert argspec.defaults is None, "Default arguments not supported in fcompute"
    assert (
        len(argspec.kwonlyargs) == 0
    ), "Keyword arguments are not supported in fcompute"

    with get_context() as ctx, get_location() as loc, Stage(name, dtype, shape) as stage:
        func_ip = InsertionPoint(get_func_body())
        hcl_mlir.set_insertion_point(func_ip)
        # create return tensor
        ret_tensor = placeholder(shape, name=name)
        ret_tensor.op = ret_tensor.op(
            ret_tensor.memref_type, None, None, None, ip=get_insertion_point()
        )

        with func_ip:
            # create stage handle
            loop_handle_type = hcl_mlir.StageHandleType.get(ctx)
            stage_handle = hcl_mlir.CreateStageHandleOp(
                loop_handle_type, StringAttr.get(name)
            )
            # update stage handle (TODO: fix this temporary method)
            stage.stage_handle = stage_handle

            # create loop handles
            loop_handles = []
            for loop_name in arg_names:
                loop_handles.append(
                    hcl_mlir.CreateLoopHandleOp(
                        hcl_mlir.LoopHandleType.get(
                            ctx), StringAttr.get(loop_name)
                    )
                )

        # create for loops in the stage
        loops = []
        body_ip = func_ip
        for i, (ub, loop_name) in enumerate(zip(shape, arg_names)):
            loop = hcl_mlir.make_constant_for(
                0,
                ub,
                step=1,
                name=loop_name,
                stage=(name if i == 0 else ""),
                ip=body_ip,
            )
            if i != 0:  # manually add terminator!
                affine.AffineYieldOp([], ip=body_ip)
            loops.append(loop)
            body_ip = InsertionPoint(loop.body)

        # transform lambda function to MLIR
        set_insertion_point(body_ip)  # inner-most loop
        # get loop variables (BlockArgument)
        iter_var = [hcl_mlir.IterVar(loop.induction_variable)
                    for loop in loops]

        # calculate the lambda funtion,
        # at the same time build up MLIR nodes;
        # the Python builtin operators are overloaded in our custom class,
        # thus fcompute can be directly called and run
        result_expr = fcompute(*iter_var)
        builder = ASTBuilder()
        true_result = builder.visit(result_expr)
        result_expr.op = true_result

        # store the result back to tensor
        # we have to read the ssa value out first, then store back to tensor
        value_attr = IntegerAttr.get(IndexType.get(), 0)
        zero_idx = std.ConstantOp(
            IndexType.get(), value_attr, ip=get_insertion_point())
        value = memref.LoadOp(
            F32Type.get(ctx),
            result_expr.op.result,
            [zero_idx.result],
            loc=loc,
            ip=get_insertion_point()
        )
        ret_val = memref.StoreOp(
            value.result,
            ret_tensor.op.result,
            [loop.induction_variable for loop in loops],
            ip=get_insertion_point(),
        )

        # remember to add affine.yield after each for loop
        affine.AffineYieldOp([], ip=get_insertion_point())

        # hard coded loop axes
        stage.mlir_axis = loop_handles

        # recover insertion point
        set_insertion_point(func_ip)

        return ret_tensor


def build_fpga_kernel(schedule, inputs, target=None, name="default_function", stmt=None):
    # generate code
    buf = io.StringIO()
    hcl_mlir.emit_hlscpp(get_module(), buf)
    buf.seek(0)
    return buf.read()


def lowerToLLVM(module):
    import mlir.conversions
    pm = passmanager.PassManager.parse(
        "reconcile-unrealized-casts")
    pm.run(module)
    return module


def build_llvm(schedule, inputs, target=None, name="default_function", stmt=None):
    with get_context() as ctx, get_location():
        # mod = get_module()
        print("\n\nBefore Lowering: ")
        get_module().dump()
        hcl_mlir.lower_hcl_to_llvm(get_module(), ctx)
        lowerToLLVM(get_module())
        print("lowered.")
        print("\n\nAfter Lowering: ")
        get_module().dump()
        execution_engine = ExecutionEngine(get_module())
        execution_engine.invoke(name)
        print("Execution success")
