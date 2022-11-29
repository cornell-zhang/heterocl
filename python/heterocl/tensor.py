import inspect
from typing import List, Callable

import hcl_mlir
import numpy as np
from hcl_mlir import ASTVisitor, GlobalInsertionPoint
from hcl_mlir.dialects import affine
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.dialects import memref
from hcl_mlir.ir import *
from hcl_mlir.exceptions import *

from .types import dtype_to_str, Int, UInt, Float, Fixed, UFixed
from .context import get_context, get_location, NestedStageLevel
from .schedule import Schedule, Stage
from .utils import get_extra_type_hints, hcl_dtype_to_mlir


class Tensor(object):
    """A wrapper class for hcl-mlir tensor-related operations
    op can be placeholder (alloc) or compute op
    """

    def __init__(self, shape, dtype, fcompute=None, name="", impl="tensor", output=None):
        # hcl.Type before building
        # hcl_mlir.Type after building
        self.dtype = dtype
        if impl == "tensor":
            self.op = hcl_mlir.TensorOp(
                shape, memref.AllocOp, dtype, name=name)
        elif impl == "compute":
            self.op = ComputeOp(shape, fcompute, dtype, name, output)
        else:
            raise RuntimeError("Not supported implementation method")
        self.uses = []
        self.name = name

        if hcl_mlir.is_build_inplace() or NestedStageLevel.get() > 0:
            self.build()

    def init(self):
        self.op.dtype = hcl_dtype_to_mlir(self.dtype)

    def build(self):
        if self.dtype is not None:
            self.init()
        self.op.build()

    def add_use(self, use):
        # `use` is another Tensor
        self.uses.append(use)

    @property
    def is_placeholder(self):
        return isinstance(self.op, hcl_mlir.TensorOp)

    @property
    def v(self):
        return self.op.v

    @v.setter
    def v(self, value):
        self.op.v = value

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
        elif key == "shape":
            return self.op.shape
        elif key == "uses":
            return self.uses
        elif key == "name":
            return self.name
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

        def get_inputs(compute_func):
            # tackle nested function problem
            closure_var = inspect.getclosurevars(compute_func).nonlocals
            for _, var in closure_var.items():
                if isinstance(var, Tensor):
                    self.inputs.append(var)
                elif isinstance(var, hcl_mlir.ReduceVar):
                    self.reduce_var.append(var)
                elif isinstance(var, Callable):
                    get_inputs(var)

        self.inputs: List[Tensor] = []
        self.reduce_var = []
        get_inputs(fcompute)

        self.shape = shape
        self.fcompute = fcompute
        self.dtype = dtype
        self.name = name
        self.stage = Stage(self.name)
        if output == None:
            if dtype == None:  # mutate
                self.kind = "mutate"
                self.output = None
            else:
                self.kind = "compute"
                self.output = Tensor(self.shape, self.dtype,
                                     name=self.name, impl="tensor")  # placeholder
        else:  # update
            self.kind = "update"
            self.output = output
        self.stage.set_output(self.output)
        self.stage.update_mapping(self.kind)
        self.arg_names = arg_names

    def build(self):
        Schedule._CurrentStage.append(self.stage)
        self.stage.stage_handle = hcl_d.CreateOpHandleOp(
            StringAttr.get(self.stage.name), ip=GlobalInsertionPoint.get()
        )
        NestedStageLevel.set(NestedStageLevel.get() + 1)
        input_types = []
        for in_tensor in self.inputs:  # hcl.Tensor -> hcl_mlir.TensorOp
            input_types.append(in_tensor.memref_type)

        # Disable build-in-place for declarative compute
        hcl_mlir.disable_build_inplace()
        # Start building loop-nest
        with get_context() as ctx, get_location() as loc:
            # create loop handles in the top function
            with GlobalInsertionPoint.get():
                loop_handles = []
                for loop_name in self.arg_names:
                    loop_handles.append(
                        hcl_d.CreateLoopHandleOp(
                            self.stage.stage_handle.result, StringAttr.get(loop_name))
                    )
                for var in self.reduce_var:
                    loop_handles.append(
                        hcl_d.CreateLoopHandleOp(self.stage.stage_handle.result, StringAttr.get(var.name)))
            # set loop handles
            if self.output is not None:
                self.stage.op.set_axis(loop_handles)
            # build output tensor
            if self.kind == "compute" and Schedule._TopFunction == None:
                self.output.build()

            # main computation part
            func_ip = GlobalInsertionPoint.get()
            # Create for loops in the stage
            loops = []
            body_ip = func_ip
            for i, (ub, loop_name) in enumerate(zip(self.shape, self.arg_names)):
                loop = hcl_mlir.make_for(
                    0,
                    ub,
                    step=1,
                    name=loop_name,
                    stage=(self.name if i == 0 else ""),
                    ip=body_ip,
                )
                loops.append(loop)
                body_ip = InsertionPoint(loop.body.operations[0])

            # transform lambda function to MLIR
            GlobalInsertionPoint.save(body_ip)  # inner-most loop
            # get loop variables (BlockArgument)
            iter_var = [hcl_mlir.IterVar(loop.induction_variable, name=loop_name)
                        for loop, loop_name in zip(loops, self.arg_names)]
            if self.output is not None:
                self.output.iter_var = iter_var

            # Calculate the lambda funtion,
            # at the same time build up MLIR nodes;
            # the Python builtin operators are overloaded in our custom class,
            # thus fcompute can be directly called and run
            if self.kind == "mutate":
                hcl_mlir.enable_build_inplace()
            # If build-in-place is enabled, generate IR as fcompute is called
            # otherwise, just build AST but do not generate IR
            result_expr = self.fcompute(*iter_var)
            if self.output is not None and result_expr is not None:
                builder = ASTVisitor()
                if isinstance(result_expr, (int, float)):
                    result_expr = hcl_mlir.ConstantOp(
                        hcl_dtype_to_mlir(self.dtype), result_expr)
                # Build IR by traversing the AST tree (from leaf to root)
                value = builder.visit(result_expr)

                # Tuple is for struct construction, in this case
                # we won't need to cast the result since it's a struct.
                if not isinstance(result_expr, tuple):
                    result_expr.built_op = value

                # Store the result to output tensor
                write_back = self.output.op.result
                elt = MemRefType(write_back.type).element_type
                write_back_elt = hcl_mlir.get_concrete_type(elt)
                if not isinstance(value, hcl_mlir.BlockArgument):
                    value = value.result
                if value.type != write_back_elt:
                    fn, ln = hcl_mlir.get_line_number(4)
                    DTypeWarning(
                        "[{} line {}], StoreOp has different input types. Cast from {} to {}.".format(
                            fn, ln, value.type, write_back_elt
                        )
                    ).warn()
                    value = hcl_mlir.CastOp(result_expr, write_back_elt)
                    value.build()
                    result = value.result
                else:
                    result = value
                ret_val = affine.AffineStoreOp(
                    result,
                    write_back,
                    [loop.induction_variable for loop in loops],
                    ip=GlobalInsertionPoint.get(),
                )
                ret_val.attributes["to"] = StringAttr.get(self.output.op.name)

            # recover insertion point from inner-most loop body
            GlobalInsertionPoint.restore()
            self.stage.set_ir_node(loops[0])
        self.stage.update_mapping(self.kind)
        if Schedule._TopFunction != None:
            hcl_mlir.enable_build_inplace()
        if self.output is not None:
            if len(self.inputs) != 0:
                Schedule._CurrentSchedule.DataflowGraph.add_edges(
                    self.inputs, self.output)
            else:  # const_tensor
                Schedule._CurrentSchedule.DataflowGraph.create_node(
                    self.output)

        NestedStageLevel.set(NestedStageLevel.get() - 1)
        Schedule._CurrentStage.pop()
        return self.output


class Array(object):
    """A wrapper class for numpy array
    Differences between array and tensor:
    tensor is only a placeholder while array holds actual values
    """

    def __init__(self, np_array, dtype):
        self.dtype = dtype  # should specify the type of `dtype`
        if isinstance(np_array, list):
            np_array = np.array(np_array)
        if dtype != None:
            # Data type check
            if isinstance(dtype, Float):
                hcl_dtype_str = dtype_to_str(dtype)
                correct_dtype = np.dtype(hcl_dtype_str)
                if np_array.dtype != correct_dtype:
                    np_array = np_array.astype(correct_dtype)
            elif isinstance(dtype, Int):
                # Handle overflow
                sb = 1 << self.dtype.bits
                sb_limit = 1 << (self.dtype.bits - 1)
                np_array = np_array % sb
                def cast_func(x): return x if x < sb_limit else x - sb
                np_array = np.vectorize(cast_func)(np_array)
                np_array = np_array.astype(np.uint64)
            elif isinstance(dtype, UInt):
                # Handle overflow
                sb = 1 << self.dtype.bits
                np_array = np_array % sb
                np_array = np_array.astype(np.uint64)
            elif isinstance(dtype, Fixed):
                # Handle overflow
                sb = 1 << self.dtype.bits
                sb_limit = 1 << (self.dtype.bits - 1)
                np_array = np_array * (2**dtype.fracs)
                np_array = np.fix(np_array) % sb
                def cast_func(x): return x if x < sb_limit else x - sb
                np_array = np.vectorize(cast_func)(np_array)
                np_array = np_array.astype(np.uint64)
            elif isinstance(dtype, UFixed):
                # Handle overflow
                sb = 1 << self.dtype.bits
                np_array = np_array * (2**dtype.fracs)
                np_array = np.fix(np_array) % sb
                np_array = np_array.astype(np.uint64)
            else:
                raise DTypeError(
                    "Type error: unrecognized type: " + str(self.dtype)
                )
        else:
            raise RuntimeError("Should provide type info")
        self.np_array = np_array

    def asnumpy(self):
        if isinstance(self.dtype, (Fixed, UFixed)):
            if isinstance(self.dtype, Fixed):
                res_array = self.np_array.astype(np.int64)
            else:
                res_array = self.np_array
            res_array = res_array.astype(
                np.float64) / float(2**(self.dtype.fracs))
            return res_array
        elif isinstance(self.dtype, Int):
            res_array = self.np_array.astype(np.int64)
            return res_array
        elif isinstance(self.dtype, Float):
            res_array = self.np_array.astype(float)
            return res_array
        else:
            return self.np_array

    def unwrap(self):
        return self.np_array
