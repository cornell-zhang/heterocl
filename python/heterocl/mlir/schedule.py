import hcl_mlir
from hcl_mlir import GlobalInsertionPoint, get_context, get_location

from mlir.dialects import builtin, std
from mlir.ir import *

from .dfg import DataflowGraph


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

        func_op = sch.get_top_function()
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
        else:
            raise RuntimeError("Function should have return value")

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
        self.device_module = Module.create(hcl_mlir.get_location())
        self.host_module = None
        self.main_func = None
        Stage._mapping = []  # operation->stage
        Schedule._IfElseStack = []
        Schedule._DataflowGraph = DataflowGraph(name, inputs)

        # create top-level function
        input_types = []
        for tensor in inputs:
            input_types.append(tensor.get_memref_type())
        with get_context() as ctx, get_location() as loc:
            func_op = builtin.FuncOp(name="top", type=FunctionType.get(
                inputs=input_types, results=[]), ip=InsertionPoint(self.device_module.body))
            func_op.add_entry_block()
            func_op.attributes["top"] = UnitAttr.get()
        GlobalInsertionPoint.save(InsertionPoint(self.device_module.body))
        GlobalInsertionPoint.save(InsertionPoint(func_op.entry_block))
        self.func_op = func_op

    def get_module(self):
        return self.device_module

    def get_top_function(self):
        return self.func_op

    def create_host_module(self):
        self.host_module = Module.create(hcl_mlir.get_location())
        # create top-level function
        with get_context() as ctx, get_location() as loc:
            self.main_func = builtin.FuncOp(name="main", type=FunctionType.get(
                inputs=[], results=[]), ip=InsertionPoint(self.host_module.body))
            self.main_func.add_entry_block()
        GlobalInsertionPoint.save(InsertionPoint(self.host_module.body))
        GlobalInsertionPoint.save(InsertionPoint(self.main_func.entry_block))
        return self.host_module

    def get_host_main_function(self):
        return self.main_func

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

    def to(self, tensor, dst=None):
        with get_context() as ctx, get_location() as loc:
            # automatically set dataflow pragma
            self.get_top_function().attributes["dataflow"] = UnitAttr.get()
            # do .to() scheduling
            to_op = hcl_mlir.ToOp(
                tensor.result, dst.stage_handle.result, ip=GlobalInsertionPoint.get())


class Stage(object):
    """A Stage represents schedule for one operation.
    """

    # TODO: Need to find a hashable way to create dict
    mapping = []  # operation->stage

    def __init__(self, name):
        # create stage handle
        with get_context() as ctx, get_location() as loc:
            loop_handle_type = hcl_mlir.StageHandleType.get(ctx)
            self.stage_handle = hcl_mlir.CreateStageHandleOp(
                loop_handle_type, StringAttr.get(name), ip=GlobalInsertionPoint.get()
            )
        # wait for setting axes
        self.loop_handles = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type is RuntimeError:
            return
        Stage._mapping.append((self.op, self))

    def set_output(self, output):
        self.op = output

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
