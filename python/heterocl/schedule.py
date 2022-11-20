import functools
import warnings

import hcl_mlir
from hcl_mlir import GlobalInsertionPoint
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.dialects import func as func_d
from hcl_mlir.ir import *
from hcl_mlir.exceptions import *

from .devices import Device, DevMemoryPair
from .context import (BreakFlag, ImperativeLoopDepth, ImperativeLoopNestCount,
                      NestedStageLevel, StageName, UniqueName, StageAttachGlobal,
                      get_context, get_location, set_context, exit_context)
from .dfg import DataflowGraph
from .utils import get_extra_type_hints, remove_moved_attr, get_src_loc
from .ir import intermediate as itmd
from .ir.intermediate import *
from .ir.ir_builder import IRBuilder
from .ir.itmd_pass import Pass, NestElseIf

# By default, Python ignores deprecation warnings.
# we have to enable it to see the warning.
warnings.simplefilter('always', DeprecationWarning)

def create_schedule_from_itmd(itmd, inputs, func, name):
    """Create a schedule from an intermediate representation.
    """
    s = Schedule(name, inputs, func)
    s._Intermediate = itmd
    
    # run passes
    nest_elif_pass = NestElseIf(s.itmd)
    nest_elif_pass.apply()
    
    set_context() # set MLIR context
    ir_builder = IRBuilder(s.itmd)
    ir_builder.build()
    exit_context() # exit MLIR context
    
    create_stage_pass = CreateStage(s.itmd, s)
    create_stage_pass.apply()

    # set device module and top func
    s._device_module = ir_builder.module
    s._device_top = s.itmd.top_func.ir_op
    s._customize_ip = InsertionPoint.at_block_terminator(s._device_top.entry_block)

    return s

def build_schedule(inputs, func=None, name=""):
    """Build a schedule for compute optimizations.
    inputs: list of Tensor
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    itmd = IR()
    itmd.top_func.args = inputs
    if func is None:
        # All operations have inserted in scope!
        outputs = list()
        for op in scope.pop():
            itmd.add_op(op)
        if len(itmd.top_func.body) == 0:
            raise APIError("received an empty algorithm specification, no operations present")
    else:
        scope.pop()
        scope.push(itmd.top_func.body)
        ret = func(*inputs)
        if ret is None:
            outputs = list()
        elif isinstance(ret, tuple):
            outputs = list(ret)
        else:
            outputs = [ret]
    itmd.top_func.return_tensors.extend(outputs)
    print(itmd)
    s = create_schedule_from_itmd(itmd, inputs, func, name)
    return s

def build_schedule_old(inputs, func=None, name=""):
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
            func_op.attributes["function_type"] = TypeAttr.get(function_type)
            otypes = "".join(
                [get_extra_type_hints(v.op.dtype) for v in outputs])
            func_op.attributes["otypes"] = StringAttr.get(otypes)

            # create block terminator
            new_outputs = []
            for output in outputs:
                new_outputs.append(output.result)
            sch.DataflowGraph.set_leaves(outputs)
            assert len(new_outputs) == len(outputs)
            ret_op = func_d.ReturnOp(
                new_outputs, ip=GlobalInsertionPoint.get())
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
            func_op.attributes["function_type"] = TypeAttr.get(function_type)
            func_op.attributes["otypes"] = StringAttr.get("")
            # create block terminator
            ret_op = func_d.ReturnOp([], ip=GlobalInsertionPoint.get())
            GlobalInsertionPoint.restore()
            GlobalInsertionPoint.save(InsertionPoint(ret_op))

    # let each stage's output be an attribute of the function
    if StageAttachGlobal.get():
        if func != None:
            func.__dict__.clear()
            for op, stage in Stage._mapping:
                if op is not None:
                    func.__setattr__(op.name, op)

    exit_context()
    remove_moved_attr(sch.device_module)
    return sch


def customize(inputs, func=None, name=""):
    try:
        return build_schedule(inputs, func, name)
    except Exception as e:
        raise e
    finally:
        # TODO: remove uneeded reset logics
        hcl_mlir.reset_build_inplace()
        NestedStageLevel.set(0)
        scope.push(list())


def create_schedule(inputs, func=None, name=""):
    """Create a schedule for compute optimizations.
    inputs: list of Tensor
    """
    return customize(inputs, func, name)


class Partition(object):
    Complete = 0
    Block = 1
    Cyclic = 2


class Schedule(object):
    """Create a compute schedule
    """
    _IfElseStack = []
    _DefFuncReturn = []
    _CurrentSchedule = None
    _CurrentStage = []
    _CurrentLoops = []  # only used in imperative DSL
    _TopFunction = None
    _ScheduleStack = []
    _CurrentIf = 0 # ptr in _IfElseStack
    _Intermediate = None

    def __init__(self, name, inputs, func=None):
        self.name = name
        self.lowered = False
        # Device-agnostic module:
        # used for transformation
        self._device_module = None
        self._device_top = None

        # Device-aware module:
        # used for generating host & xcel code
        self._host_module = None
        self._xcel_module = None
        self._host_top = None
        self._xcel_top = None
        self._host_ret = None
        self._xcel_ret = None
        self._Intermediate = None

        # Instance modules for hierarchical construction
        self._instance_modules = []

        # External module:
        # used for generating other backend codes
        self._extern_module = None
        self._extern_top = None

        # Clear stage mapping
        Stage._mapping.clear()

        # Other facilities
        Schedule._CurrentSchedule = self
        Schedule._ScheduleStack.append(self)
        Schedule._CurrentStage = []
        Schedule._CurrentLoops = []
        Schedule._TopFunction = func
        Schedule._IfElseStack = []
        Schedule._DefFuncReturn = []
        Schedule._CurrentIf = 0
        # the insertion point of customization operations
        self._customize_ip = None
        # self.DataflowGraph = DataflowGraph(name, inputs)

        # create top-level function
        # itypes = ""
        # with get_context() as ctx, get_location() as loc:
        #     input_types = []
        #     for tensor in inputs:
        #         if not isinstance(tensor.op, hcl_mlir.TensorOp):
        #             continue
        #             # raise RuntimeError("Inputs should be hcl_mlir.TensorOp")
        #         tensor.init()
        #         input_types.append(tensor.op.memref_type)
        #         itypes += get_extra_type_hints(tensor.op.dtype)
        #     device_top = func_d.FuncOp(name="top", type=FunctionType.get(
        #         inputs=input_types, results=[]), ip=InsertionPoint(self._device_module.body))
        #     device_top.attributes["itypes"] = StringAttr.get(
        #         itypes)
        #     device_top.attributes["otypes"] = StringAttr.get("")
        #     device_top.add_entry_block()
        # GlobalInsertionPoint.save(InsertionPoint(device_top))
        # GlobalInsertionPoint.save(InsertionPoint(device_top.entry_block))
        # self._device_top = device_top

    def create_host_module(self):
        set_context()
        with get_context() as ctx, get_location() as loc:
            self._host_module = Module.create(loc)
            self._host_module.operation.attributes["sym_name"] = StringAttr.get(
                "host")
            # create top-level function
            self._host_top = func_d.FuncOp(name="main", type=FunctionType.get(
                inputs=[], results=[IntegerType.get_signless(32)]), ip=InsertionPoint(self._host_module.body))
            self._host_top.add_entry_block()
            # main function return
            GlobalInsertionPoint.save(InsertionPoint(self._host_module.body))
            GlobalInsertionPoint.save(
                InsertionPoint(self._host_top.entry_block))
            ret_zero = hcl_mlir.ConstantOp(IntegerType.get_signless(32), 0)
            ret_zero.build()
            ret_op = func_d.ReturnOp(
                [ret_zero.result], ip=GlobalInsertionPoint.get())
            GlobalInsertionPoint.save(InsertionPoint(ret_op))
            self._host_ret = ret_op
        return self._host_module

    def create_xcel_module(self):
        # just a copy of the device module
        self._xcel_module = Module.parse(
            str(self._device_module), get_context())
        with get_context() as ctx:
            self._xcel_module.operation.attributes["sym_name"] = StringAttr.get(
                "xcel")
        for op in self._xcel_module.body.operations:
            if str(op.name) == "\"top\"":
                self._xcel_top = op
        for op in self._xcel_top.entry_block.operations:
            if isinstance(op, func_d.ReturnOp):
                self._xcel_ret = op
        return self._xcel_module

    def create_extern_module(self):
        set_context()
        with get_context() as ctx, get_location() as loc:
            self._extern_module = Module.create(loc)
            # create top-level function
            self._extern_top = func_d.FuncOp(name="top", type=FunctionType.get(
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

    @property
    def itmd(self):
        return self._Intermediate

    @property
    def instance_modules(self):
        return self._instance_modules

    def set_lowered(self):
        self.lowered = True

    def is_lowered(self):
        return self.lowered

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
        if self.is_lowered():
            raise APIError(".partition() must be called before lowering")
        if partition_type > 2:
            raise RuntimeError("Invalid partition type")
        if dim < 0:
            raise RuntimeError("Invalid dimension")
        if factor < 0:
            raise RuntimeError("Invalid factor")

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
            # if dim > len(MemRefType(target.type).shape):
            #     raise RuntimeError("Out-of-bound partition dimensionr. Got dim={}, but the target is of shape {}".format(dim, MemRefType(target.type).shape))
            dim = IntegerAttr.get(ui32, dim)
            res = hcl_d.PartitionOp(
                target.result, partition_kind=partition_type, dim=dim, factor=factor, ip=self._customize_ip)

    def replace(self, src, dst):
        """Replace a Tensor with another Tensor
        """
        if self.is_lowered():
            raise APIError(".replace() must be called before lowering")
        with get_context() as ctx, get_location():
            hcl_d.ReplaceOp(src.result, dst.result, ip=self._customize_ip)

    def reshape(self, target, shape):
        """Reshape a Tensor to a specified new shape
        """
        if self.is_lowered():
            raise APIError(".reshape() must be called before lowering")
        ori_size = functools.reduce(lambda a, b: a*b, target.shape, 1)
        new_size = functools.reduce(lambda a, b: a*b, shape, 1)
        if ori_size != new_size:
            raise RuntimeError(
                "The reshaped tensor should have the same total size with the original tensor")
        with get_context() as ctx, get_location():
            eletype = hcl_dtype_to_mlir(target.dtype)
            memreftype = MemRefType.get(shape, eletype)
            res = hcl_d.ReshapeOp(memreftype, target.result, ip=self._customize_ip)

    def reform(self, target, layout):
        """Change the layout of a tensor
        """
        if self.is_lowered():
            raise APIError(".reform() must be called before lowering")
        with get_context() as ctx, get_location():
            if layout == "nhwc":
                attr = AffineMap.get_permutation([0, 2, 3, 1])
            else:
                raise RuntimeError("Not supported layout")
            res = hcl_d.ReformOp(MemRefType.get(
                target.shape, target.ir_op.dtype), target.result, ip=self._customize_ip)
            res.attributes["layout"] = AffineMapAttr.get(attr)

    def reuse_at(self, target, parent, axis, name=None):
        """Create a reuse buffer reusing the output of current stage
        """
        if self.is_lowered():
            raise APIError(".reuse_at() must be called before lowering")
        if isinstance(axis, hcl_d.CreateLoopHandleOp):
            axis = axis.result
        elif isinstance(axis, OpResult):
            pass
        else:
            raise DTypeError("reuse_at() got invalid axis of type {}, please input CreateLoopHandleOp or its result".format(type(axis)))
        
        if isinstance(target, (itmd.AllocOp, hcl_d.ReuseAtOp)):
            target = target.result
        elif isinstance(target, OpResult):
            pass
        else:
            raise DTypeError("reuse_at() got invalid target of type {}, please input AllocOp or its result".format(type(target)))
        with get_context() as ctx, get_location() as loc:
            f32 = F32Type.get(ctx)
            # TODO: Need to do shape inference
            # return type of hcl_d.reuse_at op
            memref_type = MemRefType.get((1,), f32, loc=loc)
            res = hcl_d.ReuseAtOp(memref_type, target, axis, ip=self._customize_ip)
        return res

    def buffer_at(self, target, parent, axis, name=None):
        """Create a write buffer reusing the output of current stage"""
        if self.is_lowered():
            raise APIError(".buffer_at() must be called before lowering")
        with get_context() as ctx, get_location() as loc:
            i32 = IntegerType.get_signless(32)
            f32 = F32Type.get(ctx)
            # TODO: Need to do shape inference
            shape = (1,)
            memref_type = MemRefType.get(shape, f32, loc=loc)
            res = hcl_d.BufferAtOp(memref_type, target.result,
                                   axis.result, ip=self._customize_ip)
        return res

    def to(self, tensor, dst=None, fifo_depth=-1):
        if self.is_lowered():
            raise APIError(".to() must be called before lowering")
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
            try:
                tensor = tensor.tensor
            except (AttributeError, ValueError):
                try:
                    tensor = tensor._op
                except AttributeError:
                    pass
            if not isinstance(tensor, OpResult):
                tensor = tensor.result
            with get_context() as ctx, get_location() as loc:
                # automatically set dataflow pragma
                self.device_top.attributes["dataflow"] = UnitAttr.get()
                i32 = IntegerType.get_signless(32)
                fifo_depth = IntegerAttr.get(i32, fifo_depth)
                # do .to() scheduling
                to_op = hcl_d.InterKernelToOp(
                    tensor, dst.stage_handle.result, fifo_depth=fifo_depth, ip=GlobalInsertionPoint.get())

    def outline(self, *stage_list, unify=False):
        """Outline stages as a function

        e.g., s.outline([s0,s1], [s2], [s3,s4])
        """
        if self.is_lowered():
            raise APIError(".outline() must be called before lowering")
        results = []
        for i, stages in enumerate(stage_list):
            if isinstance(stages, list):
                handles = [stage.stage_handle.result for stage in stages]
                names = [stage.name for stage in stages]
            else:
                handles = [stages.stage_handle.result]
                names = [stages.name]
            with get_context() as ctx, get_location() as loc:
                op = hcl_d.OutlineOp(handles,
                                     ip=self._customize_ip)
                if unify and i > 0:
                    op.attributes["unify"] = StringAttr.get(results[0].name)
            if not unify or i == 0:
                results.append(StageFunction(names))
        return results if len(results) > 1 else results[0]


class StageFunction(object):

    def __init__(self, name=None):
        if not isinstance(name, list):
            name = [name]
        self.name = "Stage"
        for n in name:
            self.name += "_" + n
        self.module = None

    def build(self, schedule):
        set_context()
        with get_context() as ctx, get_location() as loc:
            new_module = Module.create(loc)
            # just a placeholder for inserting the function
            top = func_d.FuncOp(name="top", type=FunctionType.get(
                inputs=[], results=[]), ip=InsertionPoint(new_module.body))
            for op in schedule.device_module.body.operations:
                if str(op.name) == "\"{}\"".format(self.name):
                    op.move_before(top)
                    op.attributes["bit"] = UnitAttr.get()
                    break
            else:
                raise RuntimeError("Stage {} not found".format(self.name))
            top.operation.erase()
        self.module = new_module
        return new_module


class Stage(object):
    """A Stage represents schedule for one operation.
    """

    """ 
    obsolete note:
    Stage._mapping is a list of (Tensor, Stage) tuples
    or (Stage, Stage) tuples to keep track of all stages
    and their corresponding tensors. 
    For compute and mutate, we attach (Tensor, Stage) tuples
    For update and imperative, we attach (Stage, Stage) tuples
    """
    _mapping = []

    def __init__(self, name=None):
        if name is None:
            name = UniqueName.get("stage")
        self.name = name
        self.tensor = None
        self.stage_handle = None
        self.ip = None


    def reorder(self, *args):
        """reorder the arguments in the specified order.
        """
        if Schedule._CurrentSchedule.is_lowered():
            raise APIError(".reorder() must be called before lowering")
        args = list(args)
        for i in range(0, len(args)):
            if isinstance(args[i], int):
                args[i] = self.tensor.axis[args[i]]
            if not isinstance(args[i], OpResult):
                args[i] = args[i].result
        with get_context(), get_location():
            hcl_d.ReorderOp(args, ip=self.ip)

    def split(self, parent, factor=None, nparts=None, mode="transform"):
        """Split the stage either by factor providing outer scope, or both
        """
        if Schedule._CurrentSchedule.is_lowered():
            raise APIError(".split() must be called before lowering")
        if nparts != None or mode != "transform":
            raise RuntimeError("Not supported")
        if isinstance(parent, int):
            parent = self.tensor.axis[parent]
        if isinstance(parent, hcl_d.CreateLoopHandleOp):
            var = parent.result
        else:
            var = parent
        with get_context(), get_location():
            i32 = IntegerType.get_unsigned(32)
            factor = IntegerAttr.get(i32, factor)
            split_op = hcl_d.SplitOp(var, factor, ip=self.ip)
        return split_op.results[0], split_op.results[1]

    def tile(self, x_parent, y_parent, x_factor, y_factor):
        """ Perform tiling on two dimensions
        """
        if Schedule._CurrentSchedule.is_lowered():
            raise APIError(".tile() must be called before lowering")
        with get_context(), get_location():
            i32 = IntegerType.get_unsigned(32)
            x_factor = IntegerAttr.get(i32, x_factor)
            y_factor = IntegerAttr.get(i32, y_factor)
            if isinstance(x_parent, hcl_d.CreateLoopHandleOp):
                x_parent = x_parent.result
            if isinstance(y_parent, hcl_d.CreateLoopHandleOp):
                y_parent = y_parent.result
            tile_op = hcl_d.TileOp(
                x_parent, y_parent, x_factor, y_factor, ip=self.ip)
        return tile_op.results[0], tile_op.results[1], tile_op.results[2], tile_op.results[3]

    def pipeline(self, var, initiation_interval=1):
        """Pipeline the iteration.
        """
        if Schedule._CurrentSchedule.is_lowered():
            raise APIError(".pipeline() must be called before lowering")
        if isinstance(var, int):
            var = self.tensor.axis[var]
        if isinstance(var, hcl_d.CreateLoopHandleOp):
            var = var.result
        with get_context(), get_location():
            i32 = IntegerType.get_unsigned(32)
            ii = IntegerAttr.get(i32, initiation_interval)
            hcl_d.PipelineOp(var, ii=ii, ip=self.ip)

    def unroll(self, var, factor=0):
        """Unroll the iteration.
        """
        if Schedule._CurrentSchedule.is_lowered():
            raise APIError(".unroll() must be called before lowering")
        if isinstance(var, int):
            var = self.tensor.axis[var]
        if isinstance(var, hcl_d.CreateLoopHandleOp):
            var = var.result
        with get_context(), get_location():
            i32 = IntegerType.get_unsigned(32)
            factor = IntegerAttr.get(i32, factor)
            hcl_d.UnrollOp(var, factor=factor, ip=self.ip)

    def parallel(self, var):
        """Parallelize the iteration.
        """
        if Schedule._CurrentSchedule.is_lowered():
            raise APIError(".parallel() must be called before lowering")
        if isinstance(var, int):
            var = self.tensor.axis[var]
        with get_context(), get_location():
            hcl_d.ParallelOp(var.result, ip=self.ip)

    def fuse(self, *args):
        """Fuse multiple consecutive iteration variables into a single iteration variable.
        """
        if Schedule._CurrentSchedule.is_lowered():
            raise APIError(".fuse() must be called before lowering")
        assert len(args) >= 1, "Length of the arguments must be >=1 for fuse."
        args = list(args)
        for i in range(0, len(args)):
            if isinstance(args[i], int):
                args[i] = self.tensor.axis[args[i]]
            if not isinstance(args[i], OpResult):
                args[i] = args[i].result
        with get_context(), get_location():
            fused = hcl_d.FuseOp(args, ip=self.ip)
        return fused

    def compute_at(self, parent, axis):
        """Attach the stage at parent's scope
        """
        if Schedule._CurrentSchedule.is_lowered():
            raise APIError(".compute_at() must be called before lowering")
        if isinstance(axis, int):
            axis = parent.tensor.axis[axis]
        with get_context(), get_location():
            hcl_d.ComputeAtOp(self.stage_handle.result, parent.stage_handle.result, axis.result, ip=self.ip)

    def outline(self, axis=None, unify=None):
        """Outline a stage as a function
        """
        if Schedule._CurrentSchedule.is_lowered():
            raise APIError(".outline() must be called before lowering")
        with get_context(), get_location():
            op = hcl_d.OutlineOp([self.stage_handle.result], ip=self.ip)
            if axis is not None:
                if isinstance(axis, str):
                    op.attributes["axis"] = StringAttr.get(axis)
                else:
                    op.attributes["axis"] = axis.loop_name
            if unify is not None:
                op.attributes["unify"] = StringAttr.get(unify.name)
        if unify is not None:
            return unify
        else:
            # TODO(Niansong): think more about StageFunction
            # is it necessary?
            return StageFunction(self.name)

    def systolic(self):
        """Wrap the current stage as a systolic array
        """
        with get_context():
            self.tensor.ir_op.attributes["systolic"] = UnitAttr.get()

    def __enter__(self):
        HCLDeprecationWarning(
            "hcl.Stage() is deprecated, please remove it.").warn()

    def __exit__(self, ptype, value, trace):
        pass


class CreateStage(Pass):
    """Create HeteroCL stages, stage and loop handles.

    This pass does three things:
    1. Create stage and loop handles and set tensor.axis for all stage's tensors
    2. Attach tensors to Python functions as attributes
    3. Create a mapping from tensor to stage in Schedule
    """

    def __init__(self, intermediate, schedule):
        super().__init__("create_stage", intermediate)
        self.sch = schedule
        self.ip = InsertionPoint.at_block_terminator(self.itmd.top_func.ir_op.entry_block)

    def visit(self, op):
        self.create_stage(op)
        if hasattr(op, "body"):
            for op in op.body:
                # recursively visit the body
                self.visit(op)

    def create_stage(self, op):
        if isinstance(op, itmd.ComputeOp):
            self.create_compute_stage(op)
        else:
            pass
            # raise HCLNotImplementedError("create_stage method not implemented for op type: " + type(op))


    def create_compute_stage(self, op : itmd.ComputeOp):
        tensor = op.tensor if op.kind == "compute" else op.aux_tensor
        # Step 1: create stage and loop handles
        with get_context(), get_location():
            stage_hdl = hcl_d.CreateOpHandleOp(StringAttr.get(op.name), ip=self.ip)
            for iter_var in op.iter_vars:
                loop_hdl = hcl_d.CreateLoopHandleOp(stage_hdl.result, StringAttr.get(iter_var.name), ip=self.ip)
                tensor.axis.append(loop_hdl)
            for reduce_var in op.reduce_vars:
                loop_hdl = hcl_d.CreateLoopHandleOp(stage_hdl.result, StringAttr.get(reduce_var.name), ip=self.ip)
                tensor.axis.append(loop_hdl)

        # Step 2: attach tensors to top Python function
        top_func = Schedule._TopFunction
        if top_func is not None:
            top_func.__setattr__(tensor.name, tensor)

        # Step 3: create a mapping from tensor to stage
        stage = Stage(op.name)
        stage.ip = self.ip
        stage.tensor = tensor
        stage.stage_handle = stage_hdl
        Stage._mapping.append((tensor, stage))


    def apply(self):
        """Pass entry point"""
        top_func = self.itmd.top_func
        self.visit(top_func)