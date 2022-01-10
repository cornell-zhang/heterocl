from mlir.ir import *
import hcl_mlir
from collections import OrderedDict
import ast
import inspect
from .schedule import Stage
from .base import get_context, get_loc, get_module
from mlir.dialects import builtin, std, memref

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
    print(ret, type(ret))
    
    with Context() as ctx, Location.unknown() as loc:
        module = Module.create()
        body = module.body
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

    module = get_module()
    shape = tuple([int(s) if isinstance(s, float) else s for s in shape])
    out_ndim = len(shape)
    ret_tensor = hcl_mlir.placeholder(shape, name=name, ip=InsertionPoint(module.body))

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

    with get_context(), get_loc():
        with Stage(name, dtype, shape) as stage:
            body = module.body
            stage_handle = hcl_mlir.CreateStageHandleOp(F32Type.get(), StringAttr.get(name), ip=InsertionPoint(module.body))
            stage.stage_handle = stage_handle
            loop_handles = []
            for loop_name in arg_names:
                loop_handles.append(hcl_mlir.CreateLoopHandleOp(F32Type.get(), StringAttr.get(loop_name), ip=InsertionPoint(module.body)))

            loops = []
            for i, (ub, loop_name) in enumerate(zip(shape, arg_names)):
                loop = hcl_mlir.make_constant_for(0, ub, step=1, name=loop_name, stage=(name if i == 0 else ""), ip=InsertionPoint(body))
                loops.append(loop)
                body = loop.body

            iter_var = [hcl_mlir.IterVar(loop.induction_variable, InsertionPoint(body)) for loop in loops]
            ret = fcompute(*iter_var)
            ret_val = memref.StoreOp(ret.op.result, ret_tensor.op.result, [loop.induction_variable for loop in loops], ip=InsertionPoint(body))
            # hard coded
            stage.mlir_axis = loop_handles
            return ret_val