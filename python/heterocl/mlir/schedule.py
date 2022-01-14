import hcl_mlir
from hcl_mlir import GlobalInsertionPoint, get_context, get_location
from ordered_set import OrderedSet

from mlir.dialects import builtin, std
from mlir.ir import *

from ..schedule import Schedule, Stage
from ..tvm.schedule import create_schedule as tvm_create_schedule
from .base import get_module


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
    GlobalInsertionPoint.clear()
    # create exact HCL IR nodes
    with get_context() as ctx, get_location() as loc, Stage("_top") as top:
        # create top-level function
        input_types = []
        for tensor in inputs:
            input_types.append(tensor.memref_type)
        func_op = builtin.FuncOp(name="top", type=FunctionType.get(
            inputs=input_types, results=[]), ip=InsertionPoint(get_module().body))
        func_op.add_entry_block()
        GlobalInsertionPoint.save(InsertionPoint(func_op.entry_block))
        # create exact memref alloc
        for tensor, arg in zip(inputs, func_op.entry_block.arguments):
            tensor.op = arg
        # execute all fcompute and generate inner IR nodes
        ret = func(*inputs)

        # append the output tensors to the input list
        if ret is not None:
            if isinstance(ret, tuple):
                outputs = list(ret)
            else:
                outputs.append(ret)
        else:
            raise RuntimeError("Function should have return value")

        # recompute the function type
        return_types = [v.memref_type for v in outputs]
        function_type = FunctionType.get(
            inputs=input_types, results=return_types)
        func_op.attributes["type"] = TypeAttr.get(function_type)

        # create block terminator
        outputs = [output.op.result for output in outputs]
        ret_op = std.ReturnOp(outputs, ip=GlobalInsertionPoint.get())
        GlobalInsertionPoint.restore()

        # let the later schedule nodes insert before ret_op
        #   compute1
        #   compute2
        #   schedule1 # inserted _before_ the point
        #   ret_op    <- InsertionPoint
        GlobalInsertionPoint.save(InsertionPoint(ret_op))

    # let each stage be an attribute of the function
    for op in top.substages:
        func.__setattr__(op.name, op)
    t = Schedule.last_stages
    ops = [t_._op.op for t_ in t]
    s = Schedule(tvm_create_schedule(ops), inputs, outputs, name)
    return s
