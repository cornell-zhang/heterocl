# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import (
    APIError,
)

from ..ast import ast
from ..utils import get_src_loc
from .base import Primitive, register_primitive


@register_primitive()
class UnrollPrimitive(Primitive):
    name = "unroll"
    is_stage_primitive = True

    @staticmethod
    def apply(stage, var, factor=0):
        """Unroll the iteration."""
        from ..schedule import Schedule

        sch = Schedule._CurrentSchedule
        if sch.is_lowered():
            raise APIError(".unroll() must be called before lowering")
        if isinstance(var, int):
            var = stage.tensor.axis[var]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        unroll_op = ast.UnrollOp(var, factor, loc)
        sch.ast.top_func.body.append(unroll_op)
