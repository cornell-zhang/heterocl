from hcl_mlir.build_ir import StoreOp
from mlir.ir import *
import hcl_mlir
from collections import OrderedDict
import ast
import inspect
from .schedule import Stage
from .base import get_context, get_loc, get_module, get_function, get_func_body
from mlir.dialects import builtin, std, memref
import hcl_mlir.affine as affine
from hcl_mlir import set_insertion_point, get_insertion_point, ASTBuilder
import io

def compute_mlir(shape, fcompute, name=None, dtype=None, attrs=OrderedDict()):
    # check API correctness
    if not isinstance(shape, tuple):
        raise APIError("The shape of compute API must be a tuple")

    shape = tuple([int(s) if isinstance(s, float) else s for s in shape])
    out_ndim = len(shape)

    argspec = inspect.getfullargspec(fcompute)
    if len(argspec.args) == 0 and argspec.varargs is None:
        arg_names = ["i%d" % i for i in range(out_ndim)]
    elif argspec.varargs is not None:
        # if there is a varargs, it takes the remaining dimensions of out_ndim
        arg_names = argspec.args + \
            [f"i{i}" for i in range(out_ndim - len(argspec.args))]
    else:
        arg_names = argspec.args
        # if there are fewer args than out dimensions, the remaining dimensions
        # are implicitly broadcast
        out_ndim = len(arg_names)
    assert argspec.varkw is None, "Variable keyword arguments not supported in fcompute"
    assert argspec.defaults is None, "Default arguments not supported in fcompute"
    assert len(
        argspec.kwonlyargs) == 0, "Keyword arguments are not supported in fcompute"

    with get_context() as ctx, \
            get_loc() as loc, \
            Stage(name, dtype, shape) as stage:
        func_ip = InsertionPoint(get_func_body())
        hcl_mlir.set_insertion_point(func_ip)
        # create return tensor
        ret_tensor = hcl_mlir.placeholder(shape, name=name)

        with func_ip:
            # create stage handle
            loop_handle_type = hcl_mlir.StageHandleType.get(ctx)
            stage_handle = hcl_mlir.CreateStageHandleOp(
                loop_handle_type, StringAttr.get(name))
            # update stage handle (TODO: fix this temporary method)
            stage.stage_handle = stage_handle

            # create loop handles
            loop_handles = []
            for loop_name in arg_names:
                loop_handles.append(hcl_mlir.CreateLoopHandleOp(
                    hcl_mlir.LoopHandleType.get(ctx),
                    StringAttr.get(loop_name)))

        # create for loops in the stage
        loops = []
        body_ip = func_ip
        for i, (ub, loop_name) in enumerate(zip(shape, arg_names)):
            loop = hcl_mlir.make_constant_for(0, ub, step=1,
                                              name=loop_name,
                                              stage=(name if i == 0 else ""),
                                              ip=body_ip)
            if i != 0:  # manually add terminator!
                affine.AffineYieldOp([], ip=body_ip)
            loops.append(loop)
            body_ip = InsertionPoint(loop.body)

        # transform lambda function to MLIR
        set_insertion_point(body_ip) # inner-most loop
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
        ret_val = memref.StoreOp(result_expr.op.result, ret_tensor.op.result,
                            [loop.induction_variable for loop in loops], ip=get_insertion_point())

        # remember to add affine.yield after each for loop
        affine.AffineYieldOp([], ip=get_insertion_point())

        # hard coded loop axes
        stage.mlir_axis = loop_handles
        print(loop_handles)

        # recover insertion point
        set_insertion_point(func_ip)

        return ret_tensor


def build_hlsc(schedule, target=None, name="default_function", stmt=None):
    # block terminator
    with get_context(), get_loc():
        std.ReturnOp([], ip=get_insertion_point())
        # lowering
        func = get_function()
        # with get_module().context:
        #     pm = PassManager.parse("loop-opt")
        # pm.run(get_module())
        hcl_mlir.loop_transformation(func.operation)
        get_module().dump()
        # generate code
        buf = io.StringIO()
        hcl_mlir.emit_hlscpp(get_module(), buf)
        buf.seek(0)
    return buf.read()
