import hcl_mlir
from hcl_mlir import GlobalInsertionPoint
from hcl_mlir.dialects import builtin
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.dialects import std
from hcl_mlir.ir import *

from ..devices import Device, DevMemoryPair
from .context import (ImperativeLoopDepth, ImperativeLoopNestCount, StageName,
                      UniqueName, get_context, get_location, set_context)
from .dfg import DataflowGraph


def create_schedule(inputs, func=None, name=""):
    """Create a schedule for compute optimizations.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    # initialization
    GlobalInsertionPoint.clear()
    set_context()

    # create exact HCL IR nodes
    if name == "":
        if func != None:
            name = func.__name__
        else:
            name = UniqueName.get("schedule")
    if func == None:
        # TODO: Suppose only one output, and the output is the last argument
        output = inputs[-1]
        root_nodes = []
        for tensor in inputs:
            if isinstance(tensor.op, hcl_mlir.TensorOp):
                root_nodes.append(tensor)
        inputs = root_nodes
    sch = Schedule(name, inputs, func)

    # build IR
    with get_context() as ctx, get_location() as loc:
        # create exact IR reference
        func_op = sch.device_top
        for placeholder, arg in zip(inputs, func_op.entry_block.arguments):
            placeholder.op.update_op(arg)

        # TODO: support imperative programming
        # execute all fcompute and generate inner IR nodes
        # 1) func is hcl.compute: IR nodes not build inplace (default)
        # 2) func is defined by imperative DSL: IR nodes build inplace
        if func != None:
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
            ret = output

        if ret is not None:
            # traverse backward in AST to build IR
            def traverse(node, visited):
                if node in visited:
                    return
                visited.append(node)
                if isinstance(node.op, hcl_mlir.TensorOp):
                    if node not in inputs:
                        node.build()
                else:  # ComputeOp
                    for input in node.op.inputs:
                        traverse(input, visited)
                    node.build(sch)

            traverse(ret, [])

            outputs = []
            if isinstance(ret, tuple):
                outputs = list(ret)
            else:
                outputs.append(ret)
            # recompute the function type
            return_types = [v.memref_type for v in outputs]
            function_type = FunctionType.get(
                inputs=func_op.type.inputs, results=return_types)
            func_op.attributes["type"] = TypeAttr.get(function_type)

            # create block terminator
            new_outputs = []
            for output in outputs:
                new_outputs.append(output.result)
            sch.DataflowGraph.set_leaves(outputs)
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
    if func != None:
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
    _CurrentStage = None
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
        Schedule._CurrentStage = None
        Schedule._TopFunction = func
        Schedule._IfElseStack = []
        self.DataflowGraph = DataflowGraph(name, inputs)

        # create top-level function
        with get_context() as ctx, get_location() as loc:
            input_types = []
            for tensor in inputs:
                if not isinstance(tensor.op, hcl_mlir.TensorOp):
                    raise RuntimeError("Inputs should be hcl_mlir.TensorOp")
                tensor.init()
                input_types.append(tensor.op.memref_type)
            device_top = builtin.FuncOp(name="top", type=FunctionType.get(
                inputs=input_types, results=[]), ip=InsertionPoint(self._device_module.body))
            device_top.add_entry_block()
            if hcl_mlir.EXTRACT_FUNCTION:
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
            res = hcl_d.PartitionOp(
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
            res = hcl_d.ReuseAtOp(memref_type, parent.stage_handle.result,
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
    mapping = []  # operation->stage

    def __init__(self, name=None):
        if name is None:
            name = UniqueName.get("stage")
        self.name = name
        # create stage handle
        with get_context() as ctx, get_location() as loc:
            self.stage_handle = hcl_d.CreateStageHandleOp(
                StringAttr.get(name), ip=GlobalInsertionPoint.get()
            )
        # wait for setting axes
        self._axis = []
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

    def add_axis(self, axis):
        self._axis.append(axis)

    @property
    def axis(self):
        return self._axis

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
        var = parent
        with get_context() as ctx, get_location():
            i32 = IntegerType.get_signless(32)
            factor = IntegerAttr.get(i32, factor)
            split_op = hcl_d.SplitOp(
                self.stage_handle.result, var.result, factor, ip=GlobalInsertionPoint.get())
        return split_op.results[0], split_op.results[1]

    def tile(self, x_parent, y_parent, x_factor, y_factor):
        """ Perform tiling on two dimensions
        """
        with get_context() as ctx, get_location():
            i32 = IntegerType.get_signless(32)
            x_factor = IntegerAttr.get(i32, x_factor)
            y_factor = IntegerAttr.get(i32, y_factor)
            tile_op = hcl_d.TileOp(self.stage_handle.result, x_parent.result,
                                   y_parent.result, x_factor, y_factor, ip=GlobalInsertionPoint.get())
        return tile_op.results[0], tile_op.results[1], tile_op.results[2], tile_op.results[3]

    def pipeline(self, var, initiation_interval=1):
        """Pipeline the iteration.
        """
        if isinstance(var, int):
            var = self.op.axis[var]
        with get_context(), get_location():
            i32 = IntegerType.get_signless(32)
            ii = IntegerAttr.get(i32, initiation_interval)
            hcl_d.PipelineOp(self.stage_handle.result,
                             var.result, ii, ip=GlobalInsertionPoint.get())

    def unroll(self, var, factor=0):
        """Unroll the iteration.
        """
        if isinstance(var, int):
            var = self.op.axis[var]
        with get_context(), get_location():
            i32 = IntegerType.get_signless(32)
            factor = IntegerAttr.get(i32, factor)
            hcl_d.UnrollOp(self.stage_handle.result, var.result,
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
