# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import (
    APIError,
)

from ..ast import ast
from ..utils import get_src_loc
from .base import Primitive, register_primitive


@register_primitive()
class PipelinePrimitive(Primitive):
    name = "pipeline"
    is_stage_primitive = True

    @staticmethod
    def apply(stage, var, initiation_interval=1):
        """Pipeline the iteration."""
        from ..schedule import Schedule

        sch = Schedule._CurrentSchedule
        if sch.is_lowered():
            raise APIError(".pipeline() must be called before lowering")
        if isinstance(var, int):
            var = stage.tensor.axis[var]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        pipeline_op = ast.PipelineOp(var, initiation_interval, loc)
        sch.ast.top_func.body.append(pipeline_op)
