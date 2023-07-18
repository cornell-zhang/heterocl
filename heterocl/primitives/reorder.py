# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import (
    APIError,
)

from ..ast import ast
from ..utils import get_src_loc
from .base import Primitive, register_primitive


@register_primitive()
class ReorderPrimitive(Primitive):
    name = "reorder"
    is_stage_primitive = True

    @staticmethod
    def apply(stage, *args):
        """reorder the arguments in the specified order."""
        from ..schedule import Schedule

        sch = Schedule._CurrentSchedule
        if sch.is_lowered():
            raise APIError(".reorder() must be called before lowering")
        args = list(args)
        # pylint: disable=consider-using-enumerate
        for i in range(len(args)):
            if isinstance(args[i], int):
                args[i] = stage.tensor.axis[args[i]]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        reorder_op = ast.ReorderOp(args, loc)
        sch.ast.top_func.body.append(reorder_op)
