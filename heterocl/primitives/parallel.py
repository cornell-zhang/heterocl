# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import (
    APIError,
)

from ..ast import ast
from ..utils import get_src_loc
from .base import Primitive, register_primitive


@register_primitive()
class ParallelPrimitive(Primitive):
    name = "parallel"
    is_stage_primitive = True

    @staticmethod
    def apply(stage, var):
        """Parallelize the iteration."""
        from ..schedule import Schedule

        sch = Schedule._CurrentSchedule
        if sch.is_lowered():
            raise APIError(".parallel() must be called before lowering")
        if isinstance(var, int):
            var = stage.tensor.axis[var]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        parallel_op = ast.ParallelOp(var, loc)
        sch.ast.top_func.body.append(parallel_op)
