# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import (
    APIError,
)

from ..ast import ast
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
