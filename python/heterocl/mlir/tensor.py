import inspect
from typing import List

import hcl_mlir
import numpy as np
from hcl_mlir import ASTBuilder, GlobalInsertionPoint
from hcl_mlir.dialects import affine, arith, builtin
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.dialects import memref, std
from hcl_mlir.ir import *

from ..types import dtype_to_str, Type
from .context import get_context, get_location
from .schedule import Schedule, Stage
from .utils import hcl_dtype_to_mlir


class Tensor(object):
    """A wrapper class for hcl-mlir tensor-related operations
    op can be placeholder (alloc) or compute op
    """

    def __init__(self, shape, dtype, fcompute=None, name="", impl="tensor", output=None):
        if not isinstance(dtype, Type):
            raise RuntimeError("dtype should be hcl.Type")
        else:
            self.dtype = dtype
        if impl == "tensor":
            self.op = hcl_mlir.TensorOp(
                shape, memref.AllocOp, dtype, name=name)
        elif impl == "compute":
            self.op = ComputeOp(shape, fcompute, dtype, name, output)
        else:
            raise RuntimeError("Not supported implementation method")
        self.uses = []

        if Schedule.BUILD_INPLACE:
            self.build()

    def init(self):
        self.op.dtype = hcl_dtype_to_mlir(self.dtype)

    def build(self):
        self.init()
        self.op.build()

    def add_use(self, use):
        # `use` is another Tensor
        self.uses.append(use)

    @property
    def is_placeholder(self):
        return isinstance(self.op, hcl_mlir.TensorOp)

    @property
    def is_compute(self):
        return isinstance(self.op, ComputeOp)

    def __getattr__(self, key):
        if key == "op":
            if isinstance(self.op, hcl_mlir.TensorOp):
                return self.op
            else:  # ComputeOp
                return self.op.output.op
        elif key == "dtype":
            return self.dtype  # hcl.Type
        elif key == "uses":
            print("useafaf")
            return self.uses
        elif key == "result":
            if isinstance(self.op, hcl_mlir.TensorOp):
                return self.op.result
            else:  # ComputeOp
                return self.op.output.op.result
        else:
            if isinstance(self.op, hcl_mlir.TensorOp):
                return self.op.__getattribute__(key)
            else:  # ComputeOp
                return self.op.output.op.__getattribute__(key)

    def __getitem__(self, indices):
        if isinstance(self.op, hcl_mlir.TensorOp):
            return self.op.__getitem__(indices)
        else:  # ComputeOp
            return self.op.output.op.__getitem__(indices)

    def __setitem__(self, indices, expr):
        if isinstance(self.op, hcl_mlir.TensorOp):
            self.op.__setitem__(indices, expr)
        else:
            self.op.output.op.__setitem__(indices, expr)


class ComputeOp(object):

    def __init__(self, shape, fcompute, dtype, name, output=None):
        # check if input arguments are valid
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
            # are implicitly broadcasted
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
            if isinstance(var, Tensor):
                inputs.append(var)
        self.shape = shape
        self.fcompute = fcompute
        self.dtype = dtype
        self.name = name
        self.inputs: List[Tensor] = inputs
        if output == None:
            self.output = Tensor(shape, dtype, name=name,
                                 impl="tensor")  # placeholder
        else:  # hcl.update
            self.output = output
        self.arg_names = arg_names

    def build(self):
        input_types = []
        for in_tensor in self.inputs:  # hcl.Tensor -> hcl_mlir.TensorOp
            input_types.append(in_tensor.memref_type)

        # Disable build-in-place for declarative compute
        hcl_mlir.disable_build_inplace()
        # Start building loop-nest
        with get_context() as ctx, get_location() as loc, Stage(self.name) as stage:
            # create loop handles in the top function
            with GlobalInsertionPoint.get():
                loop_handles = []
                for loop_name in self.arg_names:
                    loop_handles.append(
                        hcl_d.CreateLoopHandleOp(StringAttr.get(loop_name))
                    )
            # build output tensor
            self.output.build()
            # main computation part
            if hcl_mlir.EXTRACT_FUNCTION:
                if self.output is not None:
                    return_types = [self.output.op.memref_type]
                else:
                    return_types = []
                # create stage function
                stage_func_name = "Stage_"+self.name
                # here we also put the return in the input argument,
                # since commonly in C++ we should pass the array by reference
                stage_func_op = builtin.FuncOp(name=stage_func_name, type=FunctionType.get(
                    inputs=input_types+return_types, results=[]), ip=GlobalInsertionPoint.ip_stack[0])
                stage_func_op.attributes["inputs"] = StringAttr.get(
                    ",".join([tensor.name for tensor in self.inputs]))
                if self.output is not None:
                    stage_func_op.attributes["outputs"] = StringAttr.get(
                        self.output.op.name)
                stage_func_op.add_entry_block()
                # attach the function to the stage
                stage.set_ir_node(stage_func_op)
                # call this function in the top function
                call_arglist = []
                for tensor in self.inputs:
                    call_arglist.append(tensor.result)
                if self.output is not None:
                    call_arglist.append(self.output.result)
                call_op = hcl_mlir.CallOp(None, stage_func_name, call_arglist)
                call_op.build()
                call_op.built_op.attributes["inputs"] = StringAttr.get(
                    ",".join([tensor.name for tensor in self.inputs]))
                if self.output is not None:
                    call_op.built_op.attributes["outputs"] = StringAttr.get(
                        self.output.op.name)
                # update inner load/store references
                # used for recovery
                original_tensor_op = []
                for tensor, arg in zip(self.inputs, stage_func_op.entry_block.arguments):
                    if isinstance(tensor.op, hcl_mlir.TensorOp):
                        original_tensor_op.append(tensor.op.built_op)
                        tensor.op.update_op(arg)
                    else:  # ComputeOp
                        original_tensor_op.append(tensor.op.output.op.built_op)
                        tensor.op.output.op.update_op(arg)
                # insertion point become the stage function inside
                GlobalInsertionPoint.save(
                    InsertionPoint(stage_func_op.entry_block))

            func_ip = GlobalInsertionPoint.get()
            # Create for loops in the stage
            loops = []
            body_ip = func_ip
            for i, (ub, loop_name) in enumerate(zip(self.shape, self.arg_names)):
                loop = hcl_mlir.make_affine_for(
                    0,
                    ub,
                    step=1,
                    name=loop_name,
                    stage=(self.name if i == 0 else ""),
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
            if self.output is not None:
                # traverse the fcompute again
                result_expr = self.fcompute(*iter_var)
                builder = ASTBuilder()
                true_result = builder.visit(result_expr)
                result_expr.built_op = true_result

                # store the result back to tensor
                # we have to read the ssa value out first, then store back to tensor
                if isinstance(result_expr, hcl_mlir.ReduceOp):
                    zero_idx = arith.ConstantOp(
                        IndexType.get(), IntegerAttr.get(IndexType.get(), 0), ip=GlobalInsertionPoint.get())
                    value = affine.AffineLoadOp(
                        result_expr.result,
                        [zero_idx.result],
                        loc=loc,
                        ip=GlobalInsertionPoint.get()
                    )
                    if isinstance(result_expr, hcl_mlir.SumOp):
                        prefix = "sum"
                    elif isinstance(result_expr, hcl_mlir.MinOp):
                        prefix = "min"
                    elif isinstance(result_expr, hcl_mlir.MaxOp):
                        prefix = "max"
                    value.attributes["from"] = StringAttr.get(
                        "{}_rv".format(prefix))
                else:
                    value = result_expr.built_op

                if hcl_mlir.EXTRACT_FUNCTION:
                    write_back = list(stage_func_op.entry_block.arguments)[-1]
                    # recover as top function op
                    for i, tensor in enumerate(self.inputs):
                        if isinstance(tensor.op, hcl_mlir.TensorOp):
                            tensor.op.update_op(original_tensor_op[i])
                        else:  # ComputeOp
                            tensor.op.output.op.update_op(
                                original_tensor_op[i])
                else:
                    write_back = self.output.op.result
                ret_val = affine.AffineStoreOp(
                    value.result,
                    write_back,
                    [loop.induction_variable for loop in loops],
                    ip=GlobalInsertionPoint.get(),
                )
                ret_val.attributes["to"] = StringAttr.get(self.output.op.name)
            else:
                self.fcompute(*iter_var)

            # remember to add affine.yield after each for loop
            affine.AffineYieldOp([], ip=GlobalInsertionPoint.get())

            # set loop handles
            if self.output is not None:
                stage.set_output(self.output)
                stage.op.set_axis(loop_handles)
            else:
                # TODO(Niansong):
                # attach axis for hcl.mutate
                pass

            # recover insertion point from inner-most loop body
            GlobalInsertionPoint.restore()

            if hcl_mlir.EXTRACT_FUNCTION:
                # recover from the subfunction
                ret_op = std.ReturnOp([], ip=GlobalInsertionPoint.get())
                GlobalInsertionPoint.restore()
            else:
                stage.set_ir_node(loops[0])

        if self.output is not None:
            hcl_mlir.enable_build_inplace()
            Schedule._CurrentSchedule.DataflowGraph.add_edges(
                self.inputs, self.output)

        return self.output


class Array(object):
    """A wrapper class for numpy array
    Differences between array and tensor:
    tensor is only a placeholder while array holds actual values
    """

    def __init__(self, np_array, dtype):
        self.dtype = dtype  # should specify the type of `dtype`
        if dtype != None:
            # Data type check
            hcl_dtype_str = dtype_to_str(dtype)
            correct_dtype = np.dtype(hcl_dtype_str)
            if np_array.dtype != correct_dtype:
                np_array = np_array.astype(correct_dtype)
        self.np_array = np_array

    def asnumpy(self):
        return self.np_array

    def unwrap(self):
        # TODO(Niansong): support unwrap fixed-point tensor here
        return self.np_array
