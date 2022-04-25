import hcl_mlir
from hcl_mlir import GlobalInsertionPoint
from hcl_mlir.dialects import builtin
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.dialects import std
from hcl_mlir.ir import *

from ..devices import Device, DevMemoryPair
from .context import (ImperativeLoopDepth, ImperativeLoopNestCount,
                      NestedCompute, StageName, UniqueName, get_context,
                      get_location, set_context)
from .dfg import DataflowGraph
from .utils import get_extra_type_hints
import functools


def build_schedule(inputs, func=None, name=""):
    """Create a schedule for compute optimizations.
    inputs: list of Tensor
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    new_inputs = []
    for tensor in inputs:
        if not isinstance(tensor.op, hcl_mlir.TensorOp) and len(tensor.op.inputs) != 0:
            raise RuntimeError("Inputs are not roots!")
        new_inputs.append(tensor)
    inputs = new_inputs
    # initialization
    GlobalInsertionPoint.clear()
    set_context()

    # create actual HCL IR nodes
    if name == "":
        if func != None:
            name = func.__name__
        else:
            name = UniqueName.get("schedule")
    sch = Schedule(name, inputs, func)

    # build IR
    with get_context() as ctx, get_location() as loc:
        # create actual IR reference
        func_op = sch.device_top
        for placeholder, arg in zip(inputs, func_op.entry_block.arguments):
            placeholder.op.update_op(arg)

        # execute all fcompute and generate inner IR nodes
        # 1) func is hcl.compute: IR nodes not build inplace (default)
        # 2) func is defined by imperative DSL: IR nodes build inplace
        hcl_mlir.flags.BIT_OP = False
        if func != None:  # can build function directly
            """
            When having code like
            def kernel(A):
                A[0][4] = 1
            It should automatically enable in-place building
            """
            hcl_mlir.enable_build_inplace()
            ret = func(*inputs)
            hcl_mlir.disable_build_inplace()
        else:
            ret = None
            # traverse forward in AST to build IR

            def topological_sort(roots):
                lst = []
                output_tensor = []
                working_set = roots.copy()
                while len(working_set) != 0:
                    node = working_set.pop(0)
                    lst.append(node)
                    if len(node.uses) == 0:  # also get the output tensors
                        output_tensor.append(node)
                    for use in node.uses:
                        flags = [
                            in_tensor in lst for in_tensor in use.op.inputs]
                        if sum(flags) == len(use.op.inputs):
                            working_set.append(use)
                return lst, output_tensor

            order, ret = topological_sort(inputs)
            # Unwrap the stage's output tensor
            # The Tensor wrapping around ComputeOp/TensorOp acts as a container
            # The ComputeOp's output Tensor is the actual returned result
            ret = [t.op.output for t in ret if not isinstance(
                t.op, hcl_mlir.TensorOp)]
            for tensor in order:
                if not isinstance(tensor.op, hcl_mlir.TensorOp):
                    tensor.build()
        if hcl_mlir.flags.BIT_OP:
            sch.device_top.attributes["bit"] = UnitAttr.get()

        if ret is not None:
            outputs = []
            if isinstance(ret, (list, tuple)):
                outputs = list(ret)
            else:
                outputs.append(ret)
            # recompute the function type
            return_types = [v.memref_type for v in outputs]
            function_type = FunctionType.get(
                inputs=func_op.type.inputs, results=return_types)
            func_op.attributes["type"] = TypeAttr.get(function_type)
            extra_otypes = "".join(
                [get_extra_type_hints(v.op.dtype) for v in outputs])
            func_op.attributes["extra_otypes"] = StringAttr.get(extra_otypes)

            # create block terminator
            new_outputs = []
            for output in outputs:
                new_outputs.append(output.result)
            try:
                sch.DataflowGraph.set_leaves(outputs)
            except:
                pass
            assert len(new_outputs) == len(outputs)
            ret_op = std.ReturnOp(new_outputs, ip=GlobalInsertionPoint.get())
            GlobalInsertionPoint.restore()

            # let the later schedule nodes insert before ret_op
            #   compute1
            #   compute2
            #   schedule1 # inserted _before_ the point
            #   ret_op    <- InsertionPoint
            GlobalInsertionPoint.save(InsertionPoint(ret_op))
        else:  # there's no return value
            function_type = FunctionType.get(
                inputs=func_op.type.inputs, results=[])
            func_op.attributes["type"] = TypeAttr.get(function_type)
            func_op.attributes["extra_otypes"] = StringAttr.get("")
            # create block terminator
            ret_op = std.ReturnOp([], ip=GlobalInsertionPoint.get())
            GlobalInsertionPoint.restore()
            GlobalInsertionPoint.save(InsertionPoint(ret_op))

    # let each stage's output be an attribute of the function
    if func != None:
        for op, stage in Stage._mapping:
            if op is not None:
                func.__setattr__(op.name, op)
    return sch


def create_schedule(inputs, func=None, name=""):
    try:
        return build_schedule(inputs, func, name)
    except Exception as e:
        raise e
    finally:
        hcl_mlir.reset_build_inplace()
        NestedCompute.set(0)


class Partition(object):
    Complete = 0
    Block = 1
    Cyclic = 2

class BlockIdx(object):
    x = 0
    y = 1
    z = 2

class ThreadIdx(object):
    x = 3
    y = 4
    z = 5

class Schedule(object):
    """Create a compute schedule
    """
    _IfElseStack = []
    _DefFuncReturn = []
    _CurrentSchedule = None
    _CurrentStage = []
    _TopFunction = None

    def __init__(self, name, inputs, func=None):
        self.name = name
        # Device-agnostic module:
        # used for transformation
        self._device_module = Module.create(get_location())
        self._device_top = None

        # Device-aware module:
        # used for generating host & xcel code
        self._host_module = None
        self._xcel_module = None
        self._host_top = None
        self._xcel_top = None

        # External module:
        # used for generating other backend codes
        self._extern_module = None
        self._extern_top = None

        # Other facilities
        Stage._mapping = []  # operation->stage
        Schedule._CurrentSchedule = self
        Schedule._CurrentStage = []
        Schedule._TopFunction = func
        Schedule._IfElseStack = []
        Schedule._DefFuncReturn = []
        self.DataflowGraph = DataflowGraph(name, inputs)

        # create top-level function
        extra_itypes = ""
        with get_context() as ctx, get_location() as loc:
            input_types = []
            for tensor in inputs:
                if not isinstance(tensor.op, hcl_mlir.TensorOp):
                    continue
                    # raise RuntimeError("Inputs should be hcl_mlir.TensorOp")
                tensor.init()
                input_types.append(tensor.op.memref_type)
                extra_itypes += get_extra_type_hints(tensor.op.dtype)
            device_top = builtin.FuncOp(name="top", type=FunctionType.get(
                inputs=input_types, results=[]), ip=InsertionPoint(self._device_module.body))
            device_top.attributes["extra_itypes"] = StringAttr.get(
                extra_itypes)
            device_top.attributes["extra_otypes"] = StringAttr.get("")
            device_top.add_entry_block()
            if hcl_mlir.is_extract_function():
                device_top.attributes["top"] = UnitAttr.get()
        GlobalInsertionPoint.save(InsertionPoint(self._device_module.body))
        GlobalInsertionPoint.save(InsertionPoint(device_top.entry_block))
        self._device_top = device_top

    def create_host_module(self):
        self._host_module = Module.create(get_location())
        with get_context() as ctx, get_location() as loc:
            # create top-level function
            self._host_top = builtin.FuncOp(name="main", type=FunctionType.get(
                inputs=[], results=[IntegerType.get_signless(32)]), ip=InsertionPoint(self._host_module.body))
            self._host_top.add_entry_block()
            # main function return
            GlobalInsertionPoint.save(InsertionPoint(self._host_module.body))
            GlobalInsertionPoint.save(
                InsertionPoint(self._host_top.entry_block))
            ret_zero = hcl_mlir.ConstantOp(IntegerType.get_signless(32), 0)
            ret_zero.build()
            ret_op = std.ReturnOp(
                [ret_zero.result], ip=GlobalInsertionPoint.get())
            GlobalInsertionPoint.save(InsertionPoint(ret_op))
        return self._host_module

    def create_xcel_module(self):
        # just a copy of the device module
        self._xcel_module = Module.parse(
            str(self._device_module), get_context())
        for op in self._xcel_module.body.operations:
            if str(op.name) == "\"top\"":
                self._xcel_top = op
        return self._xcel_module

    def create_extern_module(self):
        self._extern_module = Module.create(get_location())
        with get_context() as ctx, get_location() as loc:
            # create top-level function
            self._extern_top = builtin.FuncOp(name="top", type=FunctionType.get(
                inputs=[], results=[]), ip=InsertionPoint(self._extern_module.body))
            self._extern_top.add_entry_block()
        return self._extern_module

    @property
    def device_module(self):
        return self._device_module

    @property
    def device_top(self):
        return self._device_top

    @property
    def host_module(self):
        return self._host_module

    @property
    def host_top(self):
        return self._host_top

    @property
    def xcel_module(self):
        return self._xcel_module

    @property
    def xcel_top(self):
        return self._xcel_top

    @property
    def extern_module(self):
        return self._extern_module

    @property
    def extern_top(self):
        return self._extern_top

    def __getitem__(self, target):
        """Return a Stage
        """
        if isinstance(target, Stage):
            return target
        for op, stage in Stage._mapping:
            if op.name == target.name:
                return stage
        raise RuntimeError("Cannot find stage")

    def partition(self, target, partition_type=Partition.Complete, dim=0, factor=0):
        """Partition a Tensor into smaller Tensors or even registers
        """
        if partition_type > 2:
            raise RuntimeError("Invalid partition type")
        if dim < 0:
            raise RuntimeError("Invalid dimension")
        if factor < 0:
            raise RuntimeError("Invalid factor")
        try:
            target = target.tensor
        except (AttributeError, ValueError):
            try:
                target = target._op
            except AttributeError:
                pass
        if not isinstance(target, OpResult):
            target = target.result

        with get_context() as ctx, get_location():
            i32 = IntegerType.get_signless(32)
            if partition_type == Partition.Complete:
                partition_type = IntegerAttr.get(i32, 0)
            elif partition_type == Partition.Block:
                partition_type = IntegerAttr.get(i32, 1)
            elif partition_type == Partition.Cyclic:
                partition_type = IntegerAttr.get(i32, 2)
            else:
                raise RuntimeError("Not supported partition type")
            ui32 = IntegerType.get_unsigned(32)
            factor = IntegerAttr.get(ui32, factor)
            dim = IntegerAttr.get(ui32, dim)
            res = hcl_d.PartitionOp(
                target, partition_type, dim, factor, ip=GlobalInsertionPoint.get())

    def reshape(self, target, shape):
        """Reshape a Tensor to a specified new shape
        """
        try:
            target = target.tensor
        except (AttributeError, ValueError):
            try:
                target = target._op
            except AttributeError:
                pass
        ori_size = functools.reduce(lambda a, b: a*b, target.shape, 1)
        new_size = functools.reduce(lambda a, b: a*b, shape, 1)
        if ori_size != new_size:
            raise RuntimeError(
                "The reshaped tensor should have the same total size with the original tensor")
        with get_context() as ctx, get_location():
            res = hcl_d.ReshapeOp(MemRefType.get(
                shape, target.op.dtype), target.result, ip=GlobalInsertionPoint.get())

    def reuse_at(self, target, parent, axis, name=None):
        """Create a reuse buffer reusing the output of current stage
        """
        try:
            target = target.tensor
        except (AttributeError, ValueError):
            try:
                target = target._op
            except AttributeError:
                pass
        if not isinstance(target, OpResult):
            target = target.result
        if not isinstance(axis, OpResult):
            axis = axis.result

        with get_context() as ctx, get_location() as loc:
            i32 = IntegerType.get_signless(32)
            f32 = F32Type.get(ctx)
            # TODO: Need to do shape inference
            memref_type = MemRefType.get((1,), f32, loc=loc)
            res = hcl_d.ReuseAtOp(memref_type, parent.stage_handle.result,
                                  target, axis, ip=GlobalInsertionPoint.get())
        return res.result

    def buffer_at(self, target, parent, axis, name=None):
        """Create a write buffer reusing the output of current stage"""
        try:
            target = target.tensor
        except (AttributeError, ValueError):
            try:
                target = target._op
            except AttributeError:
                pass

        with get_context() as ctx, get_location() as loc:
            i32 = IntegerType.get_signless(32)
            f32 = F32Type.get(ctx)
            # TODO: Need to do shape inference
            memref_type = MemRefType.get(target.shape, f32, loc=loc)
            res = hcl_d.BufferAtOp(memref_type, parent.stage_handle.result,
                                   target.result, axis.result, ip=GlobalInsertionPoint.get())

    def to(self, tensor, dst=None, fifo_depth=-1):
        # host-device data movement
        if isinstance(dst, (Device, DevMemoryPair)):
            # only do annotation not mutation here
            # code change happens when building the module
            # dst.types is a str
            if not isinstance(tensor, list):
                tensor = [tensor]
            for t in tensor:
                self.DataflowGraph.propagate_annotation(t, dst.types)
        # inter-stage data movement
        elif isinstance(dst, Stage):
            with get_context() as ctx, get_location() as loc:
                # automatically set dataflow pragma
                self.device_top.attributes["dataflow"] = UnitAttr.get()
                i32 = IntegerType.get_signless(32)
                fifo_depth = IntegerAttr.get(i32, fifo_depth)
                # do .to() scheduling
                to_op = hcl_d.InterKernelToOp(
                    tensor.result, dst.stage_handle.result, fifo_depth, ip=GlobalInsertionPoint.get())


class Stage(object):
    """A Stage represents schedule for one operation.
    """

    # TODO: Need to find a hashable way to create dict
    _mapping = []  # operation->stage

    def __init__(self, name=None):
        if name is None:
            name = UniqueName.get("stage")
        self.name = name
        self.stage_handle = None
        # wait for setting axes
        self._axis = []
        StageName.set(name)
        ImperativeLoopDepth.set(0)
        ImperativeLoopNestCount.set(0)
        # auxiliary attributes
        self.op = None
        self.ir_node = None

    def done(self):
        # create stage handle
        with get_context() as ctx, get_location() as loc:
            self.stage_handle = hcl_d.CreateStageHandleOp(
                StringAttr.get(self.name), ip=GlobalInsertionPoint.get()
            )
        if self.op is None:
            # pseudo return tensor for stage with no return value
            from .operation import placeholder
            op = placeholder((1,), name=self.name)
            Stage._mapping.append((op, self))
        elif Schedule._TopFunction == None and (self.op, self) not in Stage._mapping:
            Stage._mapping.append((self.op, self))

    def add_axis(self, axis):
        self._axis.append(axis)

    @property
    def axis(self):
        return self._axis

    def set_output(self, output):
        # output: TensorOp or imperative stage
        self.op = output
        Stage._mapping.append((self.op, self))

    def set_ir_node(self, ir_node):
        self.ir_node = ir_node

    def reorder(self, *args):
        """reorder the arguments in the specified order.
        """
        args = list(args)
        for i in range(0, len(args)):
            if isinstance(args[i], int):
                args[i] = self.op.axis[args[i]]
            if not isinstance(args[i], OpResult):
                args[i] = args[i].result
        with get_context(), get_location():
            hcl_d.ReorderOp(self.stage_handle.result, args,
                            ip=GlobalInsertionPoint.get())

    def split(self, parent, factor=None, nparts=None, mode="transform"):
        """Split the stage either by factor providing outer scope, or both
        """
        if nparts != None or mode != "transform":
            raise RuntimeError("Not supported")
        if isinstance(parent, int):
            parent = self.op.axis[parent]
        idx = self.op.axis.index(parent)
        if isinstance(parent, hcl_d.CreateLoopHandleOp):
            var = parent.result
        else:
            var = parent
        with get_context() as ctx, get_location():
            i32 = IntegerType.get_unsigned(32)
            factor = IntegerAttr.get(i32, factor)
            split_op = hcl_d.SplitOp(
                self.stage_handle.result, var, factor, ip=GlobalInsertionPoint.get())
        # self.op.axis[idx] = split_op.results[0]
        # self.op.axis.insert(idx+1, split_op.results[1])
        return split_op.results[0], split_op.results[1]

    def tile(self, x_parent, y_parent, x_factor, y_factor):
        """ Perform tiling on two dimensions
        """
        idx = self.op.axis.index(x_parent)
        with get_context() as ctx, get_location():
            i32 = IntegerType.get_unsigned(32)
            x_factor = IntegerAttr.get(i32, x_factor)
            y_factor = IntegerAttr.get(i32, y_factor)
            if isinstance(x_parent, hcl_d.CreateLoopHandleOp):
                x_parent = x_parent.result
            if isinstance(y_parent, hcl_d.CreateLoopHandleOp):
                y_parent = y_parent.result
            tile_op = hcl_d.TileOp(self.stage_handle.result, x_parent,
                                   y_parent, x_factor, y_factor, ip=GlobalInsertionPoint.get())
        # self.op.axis[idx] = tile_op.results[0]
        # self.op.axis.insert(idx+1, tile_op.results[1])
        # self.op.axis.insert(idx+2, tile_op.results[2])
        # self.op.axis.insert(idx+3, tile_op.results[3])
        return tile_op.results[0], tile_op.results[1], tile_op.results[2], tile_op.results[3]
    
    def bind(self, var, thread_axis):
        assert thread_axis < 5, "cannot support NDrange with dim > 3"
        if isinstance(var, int):
            var = self.op.axis[var]
        if isinstance(var, hcl_d.CreateLoopHandleOp):
            var = var.result
        with get_context(), get_location():
            i32 = IntegerType.get_unsigned(32)
            thread_binding_type = IntegerAttr.get(i32, thread_axis)
            hcl_d.ThreadBindOp(self.stage_handle.result,
                             var, ii, ip=GlobalInsertionPoint.get())

    def pipeline(self, var, initiation_interval=1):
        """Pipeline the iteration.
        """
        if isinstance(var, int):
            var = self.op.axis[var]
        if isinstance(var, hcl_d.CreateLoopHandleOp):
            var = var.result
        with get_context(), get_location():
            i32 = IntegerType.get_unsigned(32)
            ii = IntegerAttr.get(i32, initiation_interval)
            hcl_d.PipelineOp(self.stage_handle.result,
                             var, ii, ip=GlobalInsertionPoint.get())

    def unroll(self, var, factor=0):
        """Unroll the iteration.
        """
        if isinstance(var, int):
            var = self.op.axis[var]
        if isinstance(var, hcl_d.CreateLoopHandleOp):
            var = var.result
        with get_context(), get_location():
            i32 = IntegerType.get_unsigned(32)
            factor = IntegerAttr.get(i32, factor)
            hcl_d.UnrollOp(self.stage_handle.result, var,
                           factor, ip=GlobalInsertionPoint.get())

    def parallel(self, var):
        """Parallelize the iteration.
        """
        if isinstance(var, int):
            var = self.op.axis[var]
        with get_context(), get_location():
            hcl_d.ParallelOp(self.stage_handle.result,
                             var.result, ip=GlobalInsertionPoint.get())

    def fuse(self, *args):
        """Fuse multiple consecutive iteration variables into a single iteration variable.
        """
        assert len(args) >= 1, "Length of the arguments must be >=1 for fuse."
        args = list(args)
        for i in range(0, len(args)):
            if isinstance(args[i], int):
                args[i] = self.op.axis[args[i]]
            if not isinstance(args[i], OpResult):
                args[i] = args[i].result
        with get_context() as ctx, get_location():
            loop_handle_type = hcl_d.LoopHandleType.get(ctx)
            fused = hcl_d.FuseOp(loop_handle_type, self.stage_handle.result,
                                 args, ip=GlobalInsertionPoint.get())
        return fused

    def compute_at(self, parent, scope):
        """Attach the stage at parent's scope
        """
        if isinstance(scope, int):
            scope = parent.op.axis[scope]
        with get_context() as ctx, get_location():
            compute_at = hcl_d.ComputeAtOp(
                self.stage_handle.result, parent.stage_handle.result, scope.result, ip=GlobalInsertionPoint.get())

    def systolic(self):
        """Wrap the current stage as a systolic array
        """
        with get_context() as ctx:
            self.ir_node.attributes["systolic"] = UnitAttr.get()
