import functools

from hcl_mlir.ir import *
from hcl_mlir.exceptions import *

from .devices import Device, DevMemoryPair
from .dfg import DataflowGraph
from .context import UniqueName
from .utils import get_src_loc
from .ast import ast


def _build_ast(inputs, func=None, name=""):
    """Build a schedule for compute optimizations.
    inputs: list of Tensor
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    filename, lineno = get_src_loc()
    loc = ast.Location(filename, lineno)
    top_func = ast.FuncOp("top", inputs, [], loc)
    top_func.level = 0
    if func is None:
        # All operations have been inserted in the scope already!
        outputs = list()
        for op in ast.scope.pop():
            top_func.body.append(op)
        if len(top_func.body) == 0:
            raise APIError(
                "received an empty algorithm specification, no operations present"
            )
    else:
        ast.scope.pop()
        ast.scope.push(top_func.body)
        ret = func(*inputs)
        top_func.python_callable = func
        if ret is None:
            outputs = list()
        elif isinstance(ret, tuple):
            outputs = list(ret)
        else:
            outputs = [ret]
    top_func.return_tensors.extend(outputs)
    _ast = ast.AST(top_func)
    create_stage_pass = _CreateStagesFromAST(_ast)
    create_stage_pass.apply()
    return _ast


def _build_schedule(_ast, inputs, func, name):
    """Create a schedule from an intermediate representation.
    Also used by creating schedule from scheme.
    """
    s = Schedule(name, inputs, func)
    s._ast = _ast

    # create a dataflow graph
    create_dfg_pass = _CreateDFGFromAST(_ast)
    create_dfg_pass.apply()

    s._dfg = create_dfg_pass.dfg
    return s


def _reset_builder():
    ast.scope.reset()
    Schedule._FuncDefs.clear()
    # TODO(Niansong): clear unique namer


def customize(inputs, func=None, name=""):
    try:
        _ast = _build_ast(inputs, func, name)
        s = _build_schedule(_ast, inputs, func, name)
        return s
    except Exception as e:
        raise e
    finally:
        _reset_builder()


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
    """Create a compute schedule"""

    _TopFunction = None
    _CurrentSchedule = None
    _FuncDefs = dict()

    def __init__(self, name, inputs, func=None):
        self.name = name
        self.lowered = False
        
        # MLIR modules:
        # Device-agnostic module:
        # used for transformation
        self._module = None
        self._top_func = None
        # Device-aware module:
        # used for generating host & xcel code
        self._host_module = None
        self._xcel_module = None

        # HeteroCL AST
        self._ast = None

        # Dataflow Graph
        self._dfg = None

        # Used by Stages to refer to the current schedule
        Schedule._CurrentSchedule = self
        Schedule._TopFunction = func


    @property
    def device_module(self):
        DeprecationWarning("device_module is deprecated, use module instead")
        return self._module

    @property
    def module(self):
        return self._module

    @property
    def device_top(self):
        DeprecationWarning("device_top is deprecated, use top_func instead")
        return self._top_func

    @property
    def top_func(self):
        return self._top_func

    @property
    def host_module(self):
        return self._host_module

    @property
    def xcel_module(self):
        return self._xcel_module

    @property
    def ast(self):
        return self._ast

    @property
    def DataflowGraph(self):
        return self._dfg

    def set_lowered(self):
        self.lowered = True

    def is_lowered(self):
        return self.lowered

    def __getitem__(self, target):
        """Return a Stage"""
        if isinstance(target, Stage):
            return target
        return Stage.lookup(target.name)

    def partition(self, target, partition_type=Partition.Complete, dim=0, factor=0):
        """Partition a Tensor into smaller Tensors or even registers"""
        if self.is_lowered():
            raise APIError(".partition() must be called before lowering")
        if partition_type > 2:
            raise HCLValueError("Invalid partition type")
        if dim < 0:
            raise HCLValueError("Invalid dimension")
        if factor < 0:
            raise HCLValueError("Invalid factor")

        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        if partition_type == Partition.Complete:
            partition_type = 0
        elif partition_type == Partition.Block:
            partition_type = 1
        elif partition_type == Partition.Cyclic:
            partition_type = 2
        else:
            raise HCLValueError("Not supported partition type")
        partition_op = ast.PartitionOp(target, partition_type, dim, factor, loc)
        self.ast.top_func.body.append(partition_op)

    def replace(self, src, dst):
        """Replace a Tensor with another Tensor"""
        if self.is_lowered():
            raise APIError(".replace() must be called before lowering")

        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        replace_op = ast.ReplaceOp(src, dst, loc)
        self.ast.top_func.body.append(replace_op)

    def reshape(self, target, shape):
        """Reshape a Tensor to a specified new shape"""
        if self.is_lowered():
            raise APIError(".reshape() must be called before lowering")
        ori_size = functools.reduce(lambda a, b: a * b, target.shape, 1)
        new_size = functools.reduce(lambda a, b: a * b, shape, 1)
        if ori_size != new_size:
            raise RuntimeError(
                "The reshaped tensor should have the same total size with the original tensor"
            )
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        reshape_op = ast.ReshapeOp(target, shape, loc)
        self.ast.top_func.body.append(reshape_op)

    def reform(self, target, layout):
        """Change the layout of a tensor"""
        if self.is_lowered():
            raise APIError(".reform() must be called before lowering")
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        reform_op = ast.ReformOp(target, layout, loc)
        self.ast.top_func.body.append(reform_op)

    def reuse_at(self, target, parent, axis, name=None):
        if self.is_lowered():
            raise APIError(".reuse_at() must be called before lowering")
        if not isinstance(axis, ast.LoopHandle):
            raise DTypeError(
                "reuse_at() got invalid axis of type {}".format(type(axis))
            )
        if not isinstance(target, (ast.AllocOp, ast.ReuseAtOp)):
            raise DTypeError(
                "reuse_at() got invalid target of type {}".format(type(target))
            )

        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        reuse_at_op = ast.ReuseAtOp(target, axis, loc)
        self.ast.top_func.body.append(reuse_at_op)
        return reuse_at_op

    def buffer_at(self, target, parent, axis, name=None):
        """Create a write buffer reusing the output of current stage"""
        if self.is_lowered():
            raise APIError(".buffer_at() must be called before lowering")

        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        buffer_at_op = ast.BufferAtOp(target, axis, loc)
        self.ast.top_func.body.append(buffer_at_op)
        return buffer_at_op

    def to(self, tensor, dst=None, fifo_depth=-1):
        if self.is_lowered():
            raise APIError(".to() must be called before lowering")
        # host-device data movement
        if isinstance(dst, (Device, DevMemoryPair)):
            # only do annotation not mutation here
            # code change happens when building the module
            if not isinstance(tensor, list):
                tensor = [tensor]
            for t in tensor:
                self._dfg.propagate_annotation(t, dst.types)
        # inter-stage data movement
        elif isinstance(dst, Stage):
            filename, lineno = get_src_loc()
            loc = ast.Location(filename, lineno)
            inter_kernel_to_op = ast.InterKernelToOp(tensor, dst.stage_handle, fifo_depth, loc)
            self.ast.top_func.body.append(inter_kernel_to_op)
            # outline both stages
            src = Stage.lookup(tensor.name)
            self.outline(src)
            self.outline(dst)

    def outline(self, *stage_list, unify=False):
        """Outline stages as a function

        e.g., s.outline([s0,s1], [s2], [s3,s4])
        """
        if self.is_lowered():
            raise APIError(".outline() must be called before lowering")
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        results = []
        for i, stages in enumerate(stage_list):
            if isinstance(stages, list):
                handles = [stage.stage_handle for stage in stages]
                names = [stage.name for stage in stages]
            else:
                handles = [stages.stage_handle]
                names = [stages.name]

            outline_op = ast.OutlineOp(handles, loc)
            self.ast.top_func.body.append(outline_op)
            if unify and i > 0:
                outline_op.unify = results[0].name
            else:
                results.append(StageFunction(names))
        return results if len(results) > 1 else results[0]


class StageFunction(object):
    """
    A StageFunction represents a function that is outlined
    from a stage. It is used as the return value of .outline() primitive.
    When .outline() unify is enabled, StageFunction provides a target
    function to be unified with.
    """
    def __init__(self, name):
        if not isinstance(name, list):
            name = [name]
        self.name = "Stage"
        for n in name:
            self.name += "_" + n


class Stage(object):
    """A Stage represents schedule for one operation."""

    """
    TODO(Niansong): add better comments here
    because of the syntax of HeteroCL,
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
        # Imperative stage attaches axes to Stage object
        self.axis = list()
        # Associated AST Operation
        self._ast_op = None

    @staticmethod
    def lookup(name):
        for op, stage in Stage._mapping:
            if op.name == name:
                return stage
        raise APIError("Cannot find stage: " + name)

    def reorder(self, *args):
        """reorder the arguments in the specified order."""
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".reorder() must be called before lowering")
        args = list(args)
        for i in range(0, len(args)):
            if isinstance(args[i], int):
                args[i] = self.tensor.axis[args[i]]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        reorder_op = ast.ReorderOp(args, loc)
        schedule.ast.top_func.body.append(reorder_op)

    def split(self, parent, factor=None, nparts=None, mode="transform"):
        """Split the stage either by factor providing outer scope, or both"""
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".split() must be called before lowering")
        if nparts != None or mode != "transform":
            raise HCLNotImplementedError(
                "nparts={}, mode={} not supported".format(nparts, mode)
            )
        if isinstance(parent, int):
            parent = self.tensor.axis[parent]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        split_op = ast.SplitOp(self.stage_handle, parent, factor, loc)
        schedule.ast.top_func.body.append(split_op)
        return split_op.results[0], split_op.results[1]

    def tile(self, x_parent, y_parent, x_factor, y_factor):
        """Perform tiling on two dimensions"""
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".tile() must be called before lowering")
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        tile_op = ast.TileOp(
            self.stage_handle, x_parent, y_parent, x_factor, y_factor, loc
        )
        schedule.ast.top_func.body.append(tile_op)
        return (
            tile_op.results[0],
            tile_op.results[1],
            tile_op.results[2],
            tile_op.results[3],
        )

    def pipeline(self, var, initiation_interval=1):
        """Pipeline the iteration."""
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".pipeline() must be called before lowering")
        if isinstance(var, int):
            var = self.tensor.axis[var]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        pipeline_op = ast.PipelineOp(var, initiation_interval, loc)
        schedule.ast.top_func.body.append(pipeline_op)

    def unroll(self, var, factor=0):
        """Unroll the iteration."""
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".unroll() must be called before lowering")
        if isinstance(var, int):
            var = self.tensor.axis[var]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        unroll_op = ast.UnrollOp(var, factor, loc)
        schedule.ast.top_func.body.append(unroll_op)

    def parallel(self, var):
        """Parallelize the iteration."""
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".parallel() must be called before lowering")
        if isinstance(var, int):
            var = self.tensor.axis[var]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        parallel_op = ast.ParallelOp(var, loc)
        schedule.ast.top_func.body.append(parallel_op)

    def fuse(self, *args):
        """Fuse multiple consecutive iteration variables into a single iteration variable."""
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".fuse() must be called before lowering")
        assert len(args) >= 1, "Length of the arguments must be >=1 for fuse."
        args = list(args)
        for i in range(0, len(args)):
            if isinstance(args[i], int):
                args[i] = self.tensor.axis[args[i]]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        fuse_op = ast.FuseOp(args, loc)
        schedule.ast.top_func.body.append(fuse_op)
        return fuse_op

    def compute_at(self, parent, axis):
        """Attach the stage at parent's scope"""
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".compute_at() must be called before lowering")
        if isinstance(axis, int):
            axis = parent.tensor.axis[axis]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        compute_at_op = ast.ComputeAtOp(
            self.stage_handle, parent.stage_handle, axis, loc
        )
        schedule.ast.top_func.body.append(compute_at_op)

    def outline(self, axis=None, unify=None):
        #TODO(niansong): unify
        """Outline a stage as a function"""
        schedule = Schedule._CurrentSchedule
        if schedule.is_lowered():
            raise APIError(".outline() must be called before lowering")
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        outline_op = ast.OutlineOp([self.stage_handle], loc)
        schedule.ast.top_func.body.append(outline_op)
        if axis is not None:
            if isinstance(axis, str):
                outline_op.axis = axis
            else:
                outline_op.axis = axis.loop_name
        if unify is not None:
            outline_op.unify = unify.name
            return unify
        else:
            return StageFunction(self.name)

    def systolic(self):
        """Wrap the current stage as a systolic array"""
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        systolic_op = ast.SystolicOp(self.tensor, loc)
        schedule = Schedule._CurrentSchedule
        schedule.ast.top_func.body.append(systolic_op)

    def __enter__(self):
        HCLDeprecationWarning("hcl.Stage() is deprecated, please remove it.").warn()

    def __exit__(self, ptype, value, trace):
        pass


class _CreateStagesFromAST(object):
    """Create HeteroCL stages
    This pass does three things:
    1. Create stage and loop handles and set tensor.axis for all stage's tensors
    2. Attach tensors to Python functions as attributes
    3. Create a mapping from tensor to stage in Schedule
    """

    def __init__(self, _ast):
        self._ast = _ast
        # clear the stage mapping
        Stage._mapping.clear()

    def apply(self):
        """Pass entry point"""
        top_func = self._ast.top_func
        self.visit(top_func)

    def visit(self, op):
        self.create_stage(op)
        if hasattr(op, "body") and op.body is not None:
            for op in op.body:
                # recursively visit the body
                self.visit(op)

    def create_stage(self, op):
        if isinstance(op, ast.ComputeOp):
            self.create_compute_stage(op)
        elif isinstance(op, ast.ForOp):
            self.create_imperative_stage(op)

    def create_compute_stage(self, op: ast.ComputeOp):
        # Create stage and attach attributes
        stage = Stage(op.name)
        stage._ast_op = op
        tensor = op.tensor if op.kind == "compute" else op.aux_tensor
        stage.tensor = tensor
        top_func = self._ast.top_func.python_callable
        if op.kind == "compute":
            Stage._mapping.append((tensor, stage))
            if top_func is not None:
                top_func.__setattr__(op.name, op.tensor)
        elif op.kind == "update":
            stage.__setattr__(op.tensor.name, tensor)
            Stage._mapping.append((stage, stage))
            if top_func is not None:
                top_func.__setattr__(op.name, stage)
        else: # op.kind == "mutate"
            Stage._mapping.append((stage, stage))
            if top_func is not None:
                top_func.__setattr__(op.name, stage)

        # create handles
        stage_hdl = ast.OpHandle(op.name, op.loc)
        stage.stage_handle = stage_hdl
        for iter_var in op.iter_vars + op.reduce_vars:
            loop_hdl = ast.LoopHandle(stage_hdl, iter_var.name, op.loc)
            tensor.axis.append(loop_hdl)

    def create_imperative_stage(self, op: ast.ForOp):
        if op.tag is None:
            return
        # create stage and attach attributes
        stage = Stage(op.tag)
        stage._ast_op = op
        Stage._mapping.append((stage, stage))
        top_func = self._ast.top_func.python_callable
        if top_func is not None:
            top_func.__setattr__(op.tag, stage)

        # create handles
        nested_for_loops = [op]

        def get_nested_for_loops(op):
            for body_op in op.body:
                if isinstance(body_op, ast.ForOp):
                    nested_for_loops.append(body_op)
                    get_nested_for_loops(body_op)

        get_nested_for_loops(op)
        stage_hdl = ast.OpHandle(op.tag, op.loc)
        stage.stage_handle = stage_hdl
        for loop in nested_for_loops:
            loop_hdl = ast.LoopHandle(stage_hdl, loop.name, op.loc)
            stage.axis.append(loop_hdl)
            setattr(stage, loop.name, loop_hdl)


class _CreateDFGFromAST(object):
    def __init__(self, _ast):
        self._ast = _ast
        self.dfg = DataflowGraph(name=_ast.top_func.name, inputs=_ast.top_func.args)

    def apply(self):
        """Pass entry point"""
        top_func = self._ast.top_func
        self.visit(top_func, self.create_edge)

    def visit(self, op, callback, *args, **kwargs):
        callback(op, *args, **kwargs)
        if hasattr(op, "body") and op.body is not None:
            for op in op.body:
                self.visit(op, callback, *args, **kwargs)

    def create_edge(self, op):
        if isinstance(op, ast.ComputeOp):
            if op.kind == "compute":
                for t in op.input_tensors:
                    self.dfg.add_edge(t, op.tensor)
                    # print("add edge", t, op.tensor)
            else: # update, mutate
                for t in op.input_tensors:
                    self.dfg.add_edge(t, op.aux_tensor, stateful=True)
                    # print("add edge", t, op.aux_tensor)
        elif isinstance(op, ast.ForOp):
            # raise HCLNotImplementedError("ForOp is not supported in DFG")
            pass