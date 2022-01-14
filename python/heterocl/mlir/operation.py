import inspect
from collections import OrderedDict

import hcl_mlir
import hcl_mlir.affine as affine
from hcl_mlir import (ASTBuilder, get_context,
                      get_location, GlobalInsertionPoint)

from mlir.dialects import memref, std
from mlir.ir import *

from ..schedule import Stage
from .base import get_top_function


def placeholder(shape, name=None, dtype=None):
    """Construct a HeteroCL placeholder for inputs/outputs.
    """
    with get_context() as ctx, get_location() as loc:
        memref_type = MemRefType.get(shape, F32Type.get(ctx), loc=loc)
        tensor = hcl_mlir.TensorOp(shape, memref.AllocOp, memref_type)
        return tensor


def reduce_axis(lower, upper, name=None):
    """Create a reduction axis for reduction operations.
    """
    return hcl_mlir.ReduceVar(None, bound=(lower, upper), name=name)


def sum(data, axis=None, name=""):
    return hcl_mlir.SumOp(data, axis)


def compute(shape, fcompute, name=None, dtype=None, attrs=OrderedDict()):
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
        func_ip = InsertionPoint(get_top_function().entry_block)
        # create return tensor
        ret_tensor = placeholder(shape, name=name)
        ret_tensor.op = ret_tensor.op(
            ret_tensor.memref_type, None, None, None, ip=GlobalInsertionPoint.get()
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
        GlobalInsertionPoint.save(body_ip)  # inner-most loop
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
        if isinstance(result_expr, hcl_mlir.SumOp):
            value_attr = IntegerAttr.get(IndexType.get(), 0)
            zero_idx = std.ConstantOp(
                IndexType.get(), value_attr, ip=GlobalInsertionPoint.get())
            value = memref.LoadOp(
                F32Type.get(ctx),
                result_expr.op.result,
                [zero_idx.result],
                loc=loc,
                ip=GlobalInsertionPoint.get()
            )
        else:
            value = result_expr.op
        ret_val = memref.StoreOp(
            value.result,
            ret_tensor.op.result,
            [loop.induction_variable for loop in loops],
            ip=GlobalInsertionPoint.get(),
        )

        # remember to add affine.yield after each for loop
        affine.AffineYieldOp([], ip=GlobalInsertionPoint.get())

        # hard coded loop axes
        stage.mlir_axis = loop_handles

        # recover insertion point
        GlobalInsertionPoint.restore()

        return ret_tensor
