# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument, no-name-in-module

import functools

from hcl_mlir.exceptions import (
    APIError,
    HCLDeprecationWarning,
)
from hcl_mlir.ir import (
    Location,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    UnitAttr,
    StringAttr,
)
from hcl_mlir.dialects import hcl as hcl_d
from .dfg import DataflowGraph
from .context import UniqueName
from .devices import Device, DevMemoryPair
from .utils import get_src_loc
from .ast import ast
from .ast.ir_builder import IRBuilder
from .primitives.base import PRIMITIVES, STAGE_PRIMITIVES
from .passes.pass_manager import PassManager as ast_pass_manager
from .passes.nest_if import NestElseIf
from .passes.promote_func import PromoteFunc
from .context import set_context, get_context, exit_context, get_location


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
        outputs = []
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
            outputs = []
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

    # HeteroCL Transformation Pipeline
    ast_pm = ast_pass_manager()
    ast_pm.add_pass(NestElseIf)
    ast_pm.add_pass(PromoteFunc)
    device_agnostic_ast = ast_pm.run(s.ast)
    s._ast = device_agnostic_ast
    # Build MLIR IR
    set_context()
    agnostic_ir_builder = IRBuilder(device_agnostic_ast)
    agnostic_ir_builder.build()
    s._module = agnostic_ir_builder.module
    s._top_func = agnostic_ir_builder.top_func
    exit_context()
    return s


def _reset_builder():
    ast.scope.reset()
    Schedule._FuncDefs.clear()
    UniqueName.reset()


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


class Schedule:
    """Create a compute schedule"""

    _TopFunction = None
    _CurrentSchedule = None
    _FuncDefs = {}

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

        # Register primitives.
        for pname, cls in PRIMITIVES.items():
            setattr(
                self,
                pname,
                functools.partial(
                    self.wrapped_apply, functools.partial(cls.apply, self)
                ),
            )

    def wrapped_apply(self, apply_fn, *args, **kwargs):
        filename, lineno = get_src_loc()
        with get_context(), Location.file(filename, lineno, 0):
            return apply_fn(*args, **kwargs)

    @property
    def device_module(self):
        # pylint: disable=pointless-exception-statement
        DeprecationWarning("device_module is deprecated, use module instead")
        return self._module

    @property
    def module(self):
        return self._module

    @property
    def device_top(self):
        # pylint: disable=pointless-exception-statement
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
            inter_kernel_to_op = ast.InterKernelToOp(
                tensor, dst.stage_handle, fifo_depth, loc
            )
            self.ast.top_func.body.append(inter_kernel_to_op)
            # outline both stages
            src = Stage.lookup(tensor.name)
            op = inter_kernel_to_op
            with get_context(), get_location():
                loc = Location.file(op.loc.filename, op.loc.lineno, 0)
                ir_builder = IRBuilder(self._ast)
                ip = InsertionPoint.at_block_terminator(self.top_func.entry_block)
                ir_builder.build_visitor(op.tensor, ip)
                ir_builder.build_visitor(op.stage, ip)
                i32 = IntegerType.get_signless(32)
                fifo_depth = IntegerAttr.get(i32, op.fifo_depth)
                top_func = self._ast.top_func.ir_op
                assert top_func is not None
                top_func.attributes["dataflow"] = UnitAttr.get()
                to_op = hcl_d.InterKernelToOp(
                    op.tensor.result,
                    op.stage.result,
                    fifo_depth=fifo_depth,
                    ip=ip,
                    loc=loc,
                )
                op.ir_op = to_op
            self.outline(dst)
            self.outline(src)

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
            op = outline_op
            with get_context(), get_location():
                loc = Location.file(op.loc.filename, op.loc.lineno, 0)
                ir_builder = IRBuilder(self._ast)
                ip = InsertionPoint.at_block_terminator(self.top_func.entry_block)
                for stage_hdl in op.stage_hdls:
                    ir_builder.build_visitor(stage_hdl, ip)
                hdl_results = [hdl.result for hdl in op.stage_hdls]
                hcl_outline_op = hcl_d.OutlineOp(hdl_results, ip=ip, loc=loc)
                if op.unify is not None:
                    hcl_outline_op.attributes["unify"] = StringAttr.get(op.unify)
                if op.axis is not None:
                    hcl_outline_op.attributes["axis"] = StringAttr.get(op.axis)
                op.ir_op = hcl_outline_op
        return results if len(results) > 1 else results[0]


class StageFunction:
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


class Stage:
    """A Stage represents schedule for one operation.

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
        name = UniqueName.get(name, "stage")
        self.name = name
        self.tensor = None
        self.stage_handle = None
        self.ip = None
        # Imperative stage and update stage attach axes to Stage object
        self.axis = []
        # Associated AST Operation
        self._ast_op = None

        # Register primitives.
        for pname, cls in STAGE_PRIMITIVES.items():
            setattr(
                self,
                pname,
                functools.partial(
                    self.wrapped_apply, functools.partial(cls.apply, self)
                ),
            )

    def wrapped_apply(self, apply_fn, *args, **kwargs):
        filename, lineno = get_src_loc()
        with get_context(), Location.file(filename, lineno, 0):
            return apply_fn(*args, **kwargs)

    @staticmethod
    def lookup(name):
        for op, stage in Stage._mapping:
            if op.name == name:
                return stage
        raise APIError("Cannot find stage: " + name)

    def __enter__(self):
        HCLDeprecationWarning("hcl.Stage() is deprecated, please remove it.").warn()

    def __exit__(self, ptype, value, trace):
        pass

    def outline(self, stage, axis=None, unify=None):
        """Outline a stage as a function"""
        sch = Schedule._CurrentSchedule
        if sch.is_lowered():
            raise APIError(".outline() must be called before lowering")
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        outline_op = ast.OutlineOp([stage.stage_handle], loc)
        sch.ast.top_func.body.append(outline_op)
        if axis is not None:
            if isinstance(axis, str):
                outline_op.axis = axis
            else:
                outline_op.axis = axis.loop_name
        if unify is not None:
            outline_op.unify = unify.name
            return unify
        op = outline_op
        with get_context(), get_location():
            loc = Location.file(op.loc.filename, op.loc.lineno, 0)
            ip = InsertionPoint.at_block_terminator(self.top_func.entry_block)
            for stage_hdl in op.stage_hdls:
                self.build_visitor(stage_hdl, ip)
            hdl_results = [hdl.result for hdl in op.stage_hdls]
            hcl_outline_op = hcl_d.OutlineOp(hdl_results, ip=ip, loc=loc)
            if op.unify is not None:
                hcl_outline_op.attributes["unify"] = StringAttr.get(op.unify)
            if op.axis is not None:
                hcl_outline_op.attributes["axis"] = StringAttr.get(op.axis)
            op.ir_op = hcl_outline_op
        return StageFunction(stage.name)


class _CreateStagesFromAST:
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
            # pylint: disable=redefined-argument-from-local
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
                setattr(top_func, op.name, op.tensor)
        elif op.kind == "update":
            setattr(stage, op.tensor.name, tensor)
            Stage._mapping.append((stage, stage))
            if top_func is not None:
                setattr(top_func, op.name, stage)
        else:  # op.kind == "mutate"
            Stage._mapping.append((stage, stage))
            if top_func is not None:
                setattr(top_func, op.name, stage)

        # create handles
        stage_hdl = ast.OpHandle(op.name, op.loc)
        stage.stage_handle = stage_hdl
        for iter_var in op.iter_vars + op.reduce_vars:
            loop_hdl = ast.LoopHandle(stage_hdl, iter_var.name, op.loc)
            tensor.axis.append(loop_hdl)
            stage.axis.append(loop_hdl)

    def create_imperative_stage(self, op: ast.ForOp):
        if op.tag is None:
            return
        # create stage and attach attributes
        stage = Stage(op.tag)
        stage._ast_op = op
        Stage._mapping.append((stage, stage))
        top_func = self._ast.top_func.python_callable
        if top_func is not None:
            setattr(top_func, op.tag, stage)

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


class _CreateDFGFromAST:
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
            # pylint: disable=redefined-argument-from-local
            for op in op.body:
                self.visit(op, callback, *args, **kwargs)

    def create_edge(self, op):
        if isinstance(op, ast.ComputeOp):
            if op.kind == "compute":
                for t in op.input_tensors:
                    self.dfg.add_edge(t, op.tensor)
                    # print("add edge", t, op.tensor)
            else:  # update, mutate
                for t in op.input_tensors:
                    self.dfg.add_edge(t, op.aux_tensor, stateful=True)
                    # print("add edge", t, op.aux_tensor)
        elif isinstance(op, ast.ForOp):
            # raise HCLNotImplementedError("ForOp is not supported in DFG")
            pass
