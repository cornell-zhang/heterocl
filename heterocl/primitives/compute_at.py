# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, arguments-differ

from hcl_mlir.exceptions import (
    APIError,
)
from hcl_mlir.ir import (
    Location,
    InsertionPoint,
)
from hcl_mlir.dialects import hcl as hcl_d
from ..ast import ast
from ..ast.ir_builder import IRBuilder
from ..utils import get_src_loc
from .base import Primitive, register_primitive


@register_primitive()
class ComputeAtPrimitive(Primitive):
    name = "compute_at"
    is_stage_primitive = True

    @staticmethod
    def apply(stage, parent, axis):
        """Attach the stage at parent's scope"""
        from ..schedule import Schedule

        sch = Schedule._CurrentSchedule
        if sch.is_lowered():
            raise APIError(".compute_at() must be called before lowering")
        if isinstance(axis, int):
            axis = parent.tensor.axis[axis]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        compute_at_op = ast.ComputeAtOp(
            stage.stage_handle, parent.stage_handle, axis, loc
        )
        sch.ast.top_func.body.append(compute_at_op)
        op = compute_at_op
        ir_builder = IRBuilder(sch._ast)
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        ip = InsertionPoint.at_block_terminator(sch.top_func.entry_block)
        ir_builder.build_visitor(op.stage, ip)
        ir_builder.build_visitor(op.parent, ip)
        ir_builder.build_visitor(op.axis, ip)
        hcl_compute_at_op = hcl_d.ComputeAtOp(
            op.stage.result, op.parent.result, op.axis.result, ip=ip, loc=loc
        )
        op.ir_op = hcl_compute_at_op
