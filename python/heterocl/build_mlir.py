from mlir.ir import *
import hcl_mlir
from collections import OrderedDict
import ast
import inspect
from .schedule import Stage
from .base import get_context, get_loc, get_module, get_function, get_func_body
from mlir.dialects import builtin, std, memref
import hcl_mlir.affine as affine
import io

def compute_body(name,
                lambda_ivs,
                fcompute,
                shape=(),
                dtype=None,
                tensor=None,
                attrs=OrderedDict()):

    var_list = [i.var for i in lambda_ivs]
    return_tensor = True if tensor is None else False

    ret = fcompute(*var_list)
    
    with Context() as ctx, Location.unknown() as loc:
        body = get_func_body()
        for var in lambda_ivs:
            with InsertionPoint(body):
                lb = var.dom.min.value
                ub = var.dom.min.value + var.dom.extent.value
                for_loop = hcl_mlir.make_constant_for(lb, ub, 1, name=var.var.name)
            body = for_loop.body
        module.dump()

def compute_mlir(shape, fcompute, name=None, dtype=None, attrs=OrderedDict()):
    # check API correctness
    if not isinstance(shape, tuple):
        raise APIError("The shape of compute API must be a tuple")

    shape = tuple([int(s) if isinstance(s, float) else s for s in shape])
    out_ndim = len(shape)
    ret_tensor = hcl_mlir.placeholder(shape, name=name, ip=InsertionPoint(get_func_body()))

    argspec = inspect.getfullargspec(fcompute)
    if len(argspec.args) == 0 and argspec.varargs is None:
        arg_names = ["i%d" % i for i in range(out_ndim)]
    elif argspec.varargs is not None:
        # if there is a varargs, it takes the remaining dimensions of out_ndim
        arg_names = argspec.args + [f"i{i}" for i in range(out_ndim - len(argspec.args))]
    else:
        arg_names = argspec.args
        # if there are fewer args than out dimensions, the remaining dimensions
        # are implicitly broadcast
        out_ndim = len(arg_names)
    assert argspec.varkw is None, "Variable keyword arguments not supported in fcompute"
    assert argspec.defaults is None, "Default arguments not supported in fcompute"
    assert len(argspec.kwonlyargs) == 0, "Keyword arguments are not supported in fcompute"

    with get_context() as ctx, get_loc() as loc:
        with Stage(name, dtype, shape) as stage:
            loop_handle_type = hcl_mlir.StageHandleType.get(ctx)
            stage_handle = hcl_mlir.CreateStageHandleOp(loop_handle_type, StringAttr.get(name), ip=InsertionPoint(get_func_body()))
            stage.stage_handle = stage_handle
            loop_handles = []
            for loop_name in arg_names:
                loop_handles.append(hcl_mlir.CreateLoopHandleOp(hcl_mlir.LoopHandleType.get(ctx), StringAttr.get(loop_name), ip=InsertionPoint(get_func_body())))

            loops = []
            body = get_func_body()
            for i, (ub, loop_name) in enumerate(zip(shape, arg_names)):
                loop = hcl_mlir.make_constant_for(0, ub, step=1, name=loop_name, stage=(name if i == 0 else ""), ip=InsertionPoint(body))
                if i != 0: # manually add terminator!
                    affine.AffineYieldOp([], ip=InsertionPoint(body))
                loops.append(loop)
                body = loop.body

            iter_var = [hcl_mlir.IterVar(loop.induction_variable, InsertionPoint(body)) for loop in loops]
            ret = fcompute(*iter_var)
            ret_val = memref.StoreOp(ret.op.result, ret_tensor.op.result, [loop.induction_variable for loop in loops], ip=InsertionPoint(body))
            # remember to add affine.yield after each for loop
            affine.AffineYieldOp([], ip=InsertionPoint(body))
            # hard coded
            stage.mlir_axis = loop_handles
            return ret_val

def build_hlsc(schedule, target=None, name="default_function", stmt=None):
    # block terminator
    with get_context(), get_loc():
        std.ReturnOp([],ip=InsertionPoint(get_func_body()))
    get_module().dump()
    # lowering
    func = get_function()
    # https://llvm.discourse.group/t/open-discussion-on-mlir-bindings/3159/19
    hcl_mlir.loop_transformation(func.operation)
    get_module().dump()
    # generate code
    buf = io.StringIO()
    hcl_mlir.emit_hlscpp(get_module(), buf)
    buf.seek(0)
    return buf.read()