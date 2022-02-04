import hcl_mlir

from hcl_mlir import GlobalInsertionPoint, get_context, get_location, ImperativeLoopNestCount, ImperativeLoopDepth, StageName
from heterocl.schedule import Stage

from mlir.dialects import builtin, std
from mlir.ir import *

from .dfg import DataflowGraph
from ..devices import Device, DevMemoryPair


def create_schedule(inputs, func, name=""):
    """Create a schedule for compute optimizations.
    """
    outputs = []
    if not isinstance(inputs, list):
        inputs = [inputs]
    # reset the global variables
    GlobalInsertionPoint.clear()
    # create exact HCL IR nodes
    if name == "":
        name = func.__name__
    sch = Schedule(name, inputs)
    with get_context() as ctx, get_location() as loc:

        func_op = sch.device_top
        # create exact memref alloc
        for tensor, arg in zip(inputs, func_op.entry_block.arguments):
            tensor.op = arg
        # execute all fcompute and generate inner IR nodes
        # 1) func is hcl.compute: IR nodes not build inplace (default)
        # 2) func is defined by imperative DSL: IR nodes build inplace
        hcl_mlir.enable_build_inplace()
        ret = func(*inputs)
        hcl_mlir.disable_build_inplace()

        # append the output tensors to the input list
        if ret is not None:
            if isinstance(ret, tuple):
                outputs = list(ret)
            else:
                outputs.append(ret)
            # recompute the function type
            return_types = [v.get_memref_type() for v in outputs]
            function_type = FunctionType.get(
                inputs=func_op.type.inputs, results=return_types)
            func_op.attributes["type"] = TypeAttr.get(function_type)

            # create block terminator
            new_outputs = []
            for output in outputs:
                new_outputs.append(output.result)
            Schedule._DataflowGraph.set_leaves(outputs)
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
            # create block terminator
            ret_op = std.ReturnOp([], ip=GlobalInsertionPoint.get())
            GlobalInsertionPoint.restore()
            GlobalInsertionPoint.save(InsertionPoint(ret_op))

    # let each stage's output be an attribute of the function
    for op, stage in Stage._mapping:
        func.__setattr__(op.name, op)
    return sch


class Partition(object):
    Complete = 0
    Block = 1
    Cyclic = 2


class Schedule(object):
    """Create a compute schedule
    """
    _IfElseStack = []
    _DataflowGraph = DataflowGraph()

    def __init__(self, name, inputs):
        self.name = name
        # Device-agnostic module:
        # used for transformation
        self._device_module = Module.create(hcl_mlir.get_location())
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
        Schedule._IfElseStack = []
        Schedule._DataflowGraph = DataflowGraph(name, inputs)

        # create top-level function
        input_types = []
        for tensor in inputs:
            input_types.append(tensor.get_memref_type())
        with get_context() as ctx, get_location() as loc:
            device_top = builtin.FuncOp(name="top", type=FunctionType.get(
                inputs=input_types, results=[]), ip=InsertionPoint(self._device_module.body))
            device_top.add_entry_block()
            device_top.attributes["top"] = UnitAttr.get()
        GlobalInsertionPoint.save(InsertionPoint(self._device_module.body))
        GlobalInsertionPoint.save(InsertionPoint(device_top.entry_block))
        self._device_top = device_top

    def create_host_module(self):
        self._host_module = Module.create(hcl_mlir.get_location())
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
        self._extern_module = Module.create(hcl_mlir.get_location())
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

        with get_context() as ctx, get_location():
            i32 = IntegerType.get_signless(32)
            # TODO: Change to enum type
            if partition_type == Partition.Complete:
                partition_type = IntegerAttr.get(i32, 0)
            elif partition_type == Partition.Block:
                partition_type = IntegerAttr.get(i32, 1)
            elif partition_type == Partition.Cyclic:
                partition_type = IntegerAttr.get(i32, 2)
            else:
                raise RuntimeError("Not supported partition type")
            factor = IntegerAttr.get(i32, factor)
            dim = IntegerAttr.get(i32, dim)
            res = hcl_mlir.PartitionOp(
                target.result, partition_type, dim, factor, ip=GlobalInsertionPoint.get())

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

        with get_context() as ctx, get_location() as loc:
            i32 = IntegerType.get_signless(32)
            f32 = F32Type.get(ctx)
            # TODO: Need to do shape inference
            memref_type = MemRefType.get(target.shape, f32, loc=loc)
            res = hcl_mlir.ReuseAtOp(memref_type, parent.stage_handle.result,
                                     target.result, axis.result, ip=GlobalInsertionPoint.get())

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
            res = hcl_mlir.BufferAtOp(memref_type, parent.stage_handle.result,
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
                Schedule._DataflowGraph.propagate_annotation(t, dst.types)
        # inter-stage data movement
        elif isinstance(dst, Stage):
            with get_context() as ctx, get_location() as loc:
                # automatically set dataflow pragma
                self.device_top.attributes["dataflow"] = UnitAttr.get()
                i32 = IntegerType.get_signless(32)
                fifo_depth = IntegerAttr.get(i32, fifo_depth)
                # do .to() scheduling
                to_op = hcl_mlir.InterKernelToOp(
                    tensor.result, dst.stage_handle.result, fifo_depth, ip=GlobalInsertionPoint.get())


class Stage(object):
    """A Stage represents schedule for one operation.
    """

    # TODO: Need to find a hashable way to create dict
    mapping = []  # operation->stage

    def __init__(self, name):
        self.name = name
        # create stage handle
        with get_context() as ctx, get_location() as loc:
            loop_handle_type = hcl_mlir.StageHandleType.get(ctx)
            self.stage_handle = hcl_mlir.CreateStageHandleOp(
                loop_handle_type, StringAttr.get(name), ip=GlobalInsertionPoint.get()
            )
        # wait for setting axes
        self.loop_handles = None
        StageName.set(name)
        ImperativeLoopDepth.set(0)
        ImperativeLoopNestCount.set(0)
        # auxiliary attributes
        self.op = None
        self.ir_node = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type is RuntimeError:
            return
        if ImperativeLoopNestCount.get() > 1:
            # TODO(Niansong): write a better warning message
            raise RuntimeWarning("more than one loop in ...")
        if self.op is not None:
            Stage._mapping.append((self.op, self))
        else:
            # pseudo return tensor for stage with no return value
            from .operation import placeholder
            op = placeholder((1,), name=self.name)
            Stage._mapping.append((op, self))

    def set_output(self, output):
        # output: TensorOp
        self.op = output

    def set_ir_node(self, ir_node):
        self.ir_node = ir_node

    def reorder(self, *args):
        """reorder the arguments in the specified order.
        """
        args = list(args)
        for i in range(0, len(args)):
            if isinstance(args[i], int):
                args[i] = self.axis[args[i]]
            if not isinstance(args[i], OpResult):
                args[i] = args[i].result
        with get_context(), get_location():
            hcl_mlir.ReorderOp(self.stage_handle.result, args,
                               ip=GlobalInsertionPoint.get())

    def split(self, parent, factor=None, nparts=None, mode="transform"):
        """Split the stage either by factor providing outer scope, or both
        """
        if nparts != None or mode != "transform":
            raise RuntimeError("Not supported")
        if isinstance(parent, int):
            parent = self.axis[parent]
        var = parent
        with get_context() as ctx, get_location():
            i32 = IntegerType.get_signless(32)
            factor = IntegerAttr.get(i32, factor)
            loop_handle_type = hcl_mlir.LoopHandleType.get(ctx)
            split_op = hcl_mlir.SplitOp(loop_handle_type, loop_handle_type,
                                        self.stage_handle.result, var.result, factor, ip=GlobalInsertionPoint.get())
        return split_op.results[0], split_op.results[1]

    def tile(self, x_parent, y_parent, x_factor, y_factor):
        """ Perform tiling on two dimensions
        """
        with get_context() as ctx, get_location():
            i32 = IntegerType.get_signless(32)
            x_factor = IntegerAttr.get(i32, x_factor)
            y_factor = IntegerAttr.get(i32, y_factor)
            loop_handle_type = hcl_mlir.LoopHandleType.get(ctx)
            tile_op = hcl_mlir.TileOp(loop_handle_type, loop_handle_type, loop_handle_type, loop_handle_type,
                                      self.stage_handle.result, x_parent.result, y_parent.result, x_factor, y_factor, ip=GlobalInsertionPoint.get())

    def pipeline(self, var, initiation_interval=1):
        """Pipeline the iteration.
        """
        if isinstance(var, int):
            var = self.axis[var]
        with get_context(), get_location():
            i32 = IntegerType.get_signless(32)
            ii = IntegerAttr.get(i32, initiation_interval)
            hcl_mlir.PipelineOp(self.stage_handle.result,
                                var.result, ii, ip=GlobalInsertionPoint.get())

    def unroll(self, var, factor=0):
        """Unroll the iteration.
        """
        if isinstance(var, int):
            var = self.axis[var]
        with get_context(), get_location():
            i32 = IntegerType.get_signless(32)
            factor = IntegerAttr.get(i32, factor)
            hcl_mlir.UnrollOp(self.stage_handle.result, var.result,
                              factor, ip=GlobalInsertionPoint.get())

    def parallel(self, var):
        """Parallelize the iteration.
        """
        if isinstance(var, int):
            var = self.axis[var]
        with get_context(), get_location():
            hcl_mlir.ParallelOp(self.stage_handle.result,
                                var.result, ip=GlobalInsertionPoint.get())

    def fuse(self, *args):
        """Fuse multiple consecutive iteration variables into a single iteration variable.
        """
        assert len(args) >= 1, "Length of the arguments must be >=1 for fuse."
        args = list(args)
        for i in range(0, len(args)):
            if isinstance(args[i], int):
                args[i] = self.axis[args[i]]
            if not isinstance(args[i], OpResult):
                args[i] = args[i].result
        with get_context() as ctx, get_location():
            loop_handle_type = hcl_mlir.LoopHandleType.get(ctx)
            fused = hcl_mlir.FuseOp(
                loop_handle_type, self.stage_handle.result, args, ip=GlobalInsertionPoint.get())
        return fused

    def compute_at(self, parent, scope):
        """Attach the stage at parent's scope
        """
        if isinstance(scope, int):
            scope = parent.op.axis[scope]
        with get_context() as ctx, get_location():
            loop_handle_type = hcl_mlir.LoopHandleType.get(ctx)
            fused = hcl_mlir.ComputeAtOp(
                self.stage_handle.result, parent.stage_handle.result, scope.result, ip=GlobalInsertionPoint.get())

    def systolic(self):
        """Wrap the current stage as a systolic array
        """
        with get_context() as ctx:
            self.ir_node.attributes["systolic"] = UnitAttr.get()
