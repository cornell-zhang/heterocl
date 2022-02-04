import inspect
from collections import OrderedDict

import hcl_mlir
import hcl_mlir.affine as affine
from hcl_mlir import (ASTBuilder, GlobalInsertionPoint, get_context,
                      get_location)

from mlir.dialects import memref, std, builtin
from mlir.ir import *

from .. import config, types
from .schedule import Schedule, Stage


def init(init_dtype=types.Int(32), raise_assert_exception=True):
    """Initialize a HeteroCL environment with configurations.
    """
    config.init_dtype = init_dtype
    config.raise_assert_exception = raise_assert_exception


def get_dtype_str(dtype=None):
    if not dtype is None and not isinstance(dtype, types.Type):
        raise RuntimeError("Type error")
    dtype = config.init_dtype if dtype is None else dtype
    str_type = types.dtype_to_str(dtype)
    return str_type


def placeholder(shape, name=None, dtype=None):
    """Construct a HeteroCL placeholder for inputs/outputs.
    """
    if not hcl_mlir.is_hcl_mlir_type(dtype):
        dtype = get_dtype_str(dtype)
    tensor = hcl_mlir.TensorOp(
        shape, memref.AllocOp, dtype, name=name)
    return tensor


def scalar(init, name=None, dtype=None):
    """Syntactic sugar: single-value tensor 
    - init: int, float, or expr
    """
    ret_tensor = placeholder((1,), name=name, dtype=dtype)
    ret_tensor.build()
    index = hcl_mlir.ConstantOp(hcl_mlir.idx_type, 0)
    if not hcl_mlir.is_hcl_mlir_type(dtype):
        dtype = get_dtype_str(dtype)
    if isinstance(init, int) or isinstance(init, float):
        init = hcl_mlir.ConstantOp(dtype, init)
    hcl_mlir.StoreOp(init, ret_tensor, [index])
    return ret_tensor


def reduce_axis(lower, upper, name=None):
    """Create a reduction axis for reduction operations.
    """
    return hcl_mlir.ReduceVar(None, bound=(lower, upper), name=name)


def sum(data, axis=None, dtype=None, name=""):
    return hcl_mlir.SumOp(data, axis, get_dtype_str(dtype))


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

    # get input tensors
    closure_var = inspect.getclosurevars(fcompute).nonlocals
    inputs = []
    for _, var in closure_var.items():
        if isinstance(var, hcl_mlir.TensorOp):
            inputs.append(var)
    input_types = []
    for tensor in inputs:
        input_types.append(tensor.get_memref_type())

    hcl_mlir.disable_build_inplace()
    with get_context() as ctx, get_location() as loc, Stage(name) as stage:
        # create return tensor
        ret_tensor = placeholder(shape, dtype=dtype, name=name)
        # build return tensor (outside the inner function)
        ret_tensor.build()

        # create loop handles in the top function
        with GlobalInsertionPoint.get():
            loop_handles = []
            for loop_name in arg_names:
                loop_handles.append(
                    hcl_mlir.CreateLoopHandleOp(
                        hcl_mlir.LoopHandleType.get(
                            ctx), StringAttr.get(loop_name)
                    )
                )

        if hcl_mlir.EXTRACT_FUNCTION:
            return_types = [ret_tensor.get_memref_type()]
            # create stage function
            stage_func_name = "Stage_"+name
            # here we also put the return in the input argument,
            # since commonly in C++ we should pass the array by reference
            stage_func_op = builtin.FuncOp(name=stage_func_name, type=FunctionType.get(
                inputs=input_types+return_types, results=[]), ip=GlobalInsertionPoint.ip_stack[0])
            stage_func_op.attributes["inputs"] = StringAttr.get(
                ",".join([tensor.name for tensor in inputs]))
            stage_func_op.attributes["outputs"] = StringAttr.get(
                ret_tensor.name)
            stage_func_op.add_entry_block()
            # call this function in the top function
            call_op = hcl_mlir.CallOp(None, stage_func_name, [
                                      tensor.result for tensor in inputs]+[ret_tensor.result])
            call_op.build()
            call_op.built_op.attributes["inputs"] = StringAttr.get(
                ",".join([tensor.name for tensor in inputs]))
            call_op.built_op.attributes["outputs"] = StringAttr.get(
                ret_tensor.name)
            # update inner load/store references
            # used for recovery
            original_tensor_op = [tensor.op for tensor in inputs]
            for tensor, arg in zip(inputs, stage_func_op.entry_block.arguments):
                tensor.op = arg
            # insertion point become the stage function inside
            GlobalInsertionPoint.save(
                InsertionPoint(stage_func_op.entry_block))

        func_ip = GlobalInsertionPoint.get()

        # create for loops in the stage
        loops = []
        body_ip = func_ip
        for i, (ub, loop_name) in enumerate(zip(shape, arg_names)):
            loop = hcl_mlir.make_affine_for(
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
        result_expr.built_op = true_result

        # store the result back to tensor
        # we have to read the ssa value out first, then store back to tensor
        if isinstance(result_expr, hcl_mlir.SumOp):
            zero_idx = std.ConstantOp(
                IndexType.get(), IntegerAttr.get(IndexType.get(), 0), ip=GlobalInsertionPoint.get())
            value = affine.AffineLoadOp(
                hcl_mlir.get_mlir_type(result_expr.dtype),
                result_expr.result,
                [zero_idx.result],
                loc=loc,
                ip=GlobalInsertionPoint.get()
            )
            value.attributes["from"] = StringAttr.get("sum_rv")
        else:
            value = result_expr.built_op

        if hcl_mlir.EXTRACT_FUNCTION:
            write_back = list(stage_func_op.entry_block.arguments)[-1]
            # recover as top function op
            for i, tensor in enumerate(inputs):
                tensor.op = original_tensor_op[i]
        else:
            write_back = ret_tensor.result
        ret_val = affine.AffineStoreOp(
            value.result,
            write_back,
            [loop.induction_variable for loop in loops],
            ip=GlobalInsertionPoint.get(),
        )
        ret_val.attributes["to"] = StringAttr.get(ret_tensor.name)

        # remember to add affine.yield after each for loop
        affine.AffineYieldOp([], ip=GlobalInsertionPoint.get())

        # set loop handles
        stage.set_output(ret_tensor)
        stage.op.set_axis(loop_handles)

        # recover insertion point from inner-most loop body
        GlobalInsertionPoint.restore()

        if hcl_mlir.EXTRACT_FUNCTION:
            # recover from the subfunction
            ret_op = std.ReturnOp([], ip=GlobalInsertionPoint.get())
            GlobalInsertionPoint.restore()

    hcl_mlir.enable_build_inplace()
    Schedule._DataflowGraph.add_edges(inputs, ret_tensor)
    return ret_tensor


def update(tensor, fcompute, name=None):
    """
    tensor: hcl_mlir.build_ir.TensorOp
    fcompute: function, callable
    name: str
    """
    # Check tensor type
    if not isinstance(tensor, hcl_mlir.build_ir.TensorOp):
        raise RuntimeError("Unexpected argument type of the " +
                           "first argument: {}, update API expects tensor as input.".format(type(tensor)))
    shape = tensor.shape
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
    # Get input tensors to fcompute
    closure_var = inspect.getclosurevars(fcompute).nonlocals
    inputs = []
    for _, var in closure_var.items():
        if isinstance(var, hcl_mlir.TensorOp):
            inputs.append(var)
    input_types = []
    for in_tensor in inputs:
        input_types.append(in_tensor.get_memref_type())

    # Disable build-in-place for update
    hcl_mlir.disable_build_inplace()
    # Start building loop-nest
    with get_context() as ctx, get_location() as loc, Stage(name) as stage:
        # create loop handles in the top function
        with GlobalInsertionPoint.get():
            loop_handles = []
            for loop_name in arg_names:
                loop_handles.append(
                    hcl_mlir.CreateLoopHandleOp(
                        hcl_mlir.LoopHandleType.get(
                            ctx), StringAttr.get(loop_name)
                    )
                )
        if hcl_mlir.EXTRACT_FUNCTION:
            return_types = [tensor.get_memref_type()]
            # create stage function
            stage_func_name = "Stage_"+name
            # here we also put the return in the input argument,
            # since commonly in C++ we should pass the array by reference
            stage_func_op = builtin.FuncOp(name=stage_func_name, type=FunctionType.get(
                inputs=input_types+return_types, results=[]), ip=GlobalInsertionPoint.ip_stack[0])
            stage_func_op.attributes["inputs"] = StringAttr.get(
                ",".join([tensor.name for tensor in inputs]))
            stage_func_op.attributes["outputs"] = StringAttr.get(
                tensor.name)
            stage_func_op.add_entry_block()
            # call this function in the top function
            call_op = hcl_mlir.CallOp(None, stage_func_name, [
                                      tensor.result for tensor in inputs]+[tensor.result])
            call_op.build()
            call_op.built_op.attributes["inputs"] = StringAttr.get(
                ",".join([tensor.name for tensor in inputs]))
            call_op.built_op.attributes["outputs"] = StringAttr.get(
                tensor.name)
            # update inner load/store references
            # used for recovery
            original_tensor_op = [tensor.op for tensor in inputs]
            for tensor, arg in zip(inputs, stage_func_op.entry_block.arguments):
                tensor.op = arg
            # insertion point become the stage function inside
            GlobalInsertionPoint.save(
                InsertionPoint(stage_func_op.entry_block))

        func_ip = GlobalInsertionPoint.get()
        # Create for loops in the stage
        loops = []
        body_ip = func_ip
        for i, (ub, loop_name) in enumerate(zip(shape, arg_names)):
            loop = hcl_mlir.make_affine_for(
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
        result_expr.built_op = true_result
        # store the result back to tensor
        # we have to read the ssa value out first, then store back to tensor
        if isinstance(result_expr, hcl_mlir.SumOp):
            zero_idx = std.ConstantOp(
                IndexType.get(), IntegerAttr.get(IndexType.get(), 0), ip=GlobalInsertionPoint.get())
            value = affine.AffineLoadOp(
                hcl_mlir.get_mlir_type(result_expr.dtype),
                result_expr.result,
                [zero_idx.result],
                loc=loc,
                ip=GlobalInsertionPoint.get()
            )
            value.attributes["from"] = StringAttr.get("sum_rv")
        else:
            value = result_expr.built_op

        if hcl_mlir.EXTRACT_FUNCTION:
            write_back = list(stage_func_op.entry_block.arguments)[-1]
            # recover as top function op
            for i, tensor in enumerate(inputs):
                tensor.op = original_tensor_op[i]
        else:
            write_back = tensor.result
        ret_val = affine.AffineStoreOp(
            value.result,
            write_back,
            [loop.induction_variable for loop in loops],
            ip=GlobalInsertionPoint.get(),
        )
        ret_val.attributes["to"] = StringAttr.get(tensor.name)

        # remember to add affine.yield after each for loop
        affine.AffineYieldOp([], ip=GlobalInsertionPoint.get())

        # set loop handles
        stage.set_output(tensor)
        stage.op.set_axis(loop_handles)

        # recover insertion point from inner-most loop body
        GlobalInsertionPoint.restore()

        if hcl_mlir.EXTRACT_FUNCTION:
            # recover from the subfunction
            ret_op = std.ReturnOp([], ip=GlobalInsertionPoint.get())
            GlobalInsertionPoint.restore()

    hcl_mlir.enable_build_inplace()
    Schedule._DataflowGraph.add_edges(inputs, tensor)
    return


def mutate(domain, fcompute, name):
    """
    For now, assume no return value
    """
    # TODO(Niansong): sanity check
    out_ndim = len(domain)
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

    # get input tensors
    closure_var = inspect.getclosurevars(fcompute).nonlocals
    inputs = []
    for _, var in closure_var.items():
        if isinstance(var, hcl_mlir.TensorOp):
            inputs.append(var)
    input_types = []
    for tensor in inputs:
        input_types.append(tensor.get_memref_type())

    hcl_mlir.disable_build_inplace()
    with get_context() as ctx, get_location() as loc, Stage(name) as stage:
        # create return tensor
        # ret_tensor = placeholder(shape, dtype=dtype, name=name)
        # build return tensor (outside the inner function)
        # ret_tensor.build()

        # create loop handles in the top function
        with GlobalInsertionPoint.get():
            loop_handles = []
            for loop_name in arg_names:
                loop_handles.append(
                    hcl_mlir.CreateLoopHandleOp(
                        hcl_mlir.LoopHandleType.get(
                            ctx), StringAttr.get(loop_name)
                    )
                )

        func_ip = GlobalInsertionPoint.get()

        # create for loops in the stage
        loops = []
        body_ip = func_ip
        for i, (ub, loop_name) in enumerate(zip(domain, arg_names)):
            loop = hcl_mlir.make_affine_for(
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
        # builder = ASTBuilder()
        # true_result = builder.visit(result_expr)
        # result_expr.built_op = true_result

        # # store the result back to tensor
        # # we have to read the ssa value out first, then store back to tensor
        # if isinstance(result_expr, hcl_mlir.SumOp):
        #     zero_idx = std.ConstantOp(
        #         IndexType.get(), IntegerAttr.get(IndexType.get(), 0), ip=GlobalInsertionPoint.get())
        #     value = affine.AffineLoadOp(
        #         hcl_mlir.get_mlir_type(result_expr.dtype),
        #         result_expr.result,
        #         [zero_idx.result],
        #         loc=loc,
        #         ip=GlobalInsertionPoint.get()
        #     )
        #     value.attributes["from"] = StringAttr.get("sum_rv")
        # else:
        #     value = result_expr.built_op

        # if hcl_mlir.EXTRACT_FUNCTION:
        #     write_back = list(stage_func_op.entry_block.arguments)[-1]
        #     # recover as top function op
        #     for i, tensor in enumerate(inputs):
        #         tensor.op = original_tensor_op[i]
        # else:
        #     write_back = ret_tensor.result
        # ret_val = affine.AffineStoreOp(
        #     value.result,
        #     write_back,
        #     [loop.induction_variable for loop in loops],
        #     ip=GlobalInsertionPoint.get(),
        # )
        # ret_val.attributes["to"] = StringAttr.get(ret_tensor.name)

        # remember to add affine.yield after each for loop
        affine.AffineYieldOp([], ip=GlobalInsertionPoint.get())

        # set loop handles
        # stage.set_output(ret_tensor)
        # stage.op.set_axis(loop_handles)

        # recover insertion point from inner-most loop body
        GlobalInsertionPoint.restore()

        # if hcl_mlir.EXTRACT_FUNCTION:
        #     # recover from the subfunction
        #     ret_op = std.ReturnOp([], ip=GlobalInsertionPoint.get())
        #     GlobalInsertionPoint.restore()

    # hcl_mlir.enable_build_inplace()
    # Schedule._DataflowGraph.add_edges(inputs, ret_tensor)
    return
