# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import (
    APIError,
    HCLNotImplementedError,
)

from ..ast import ast
from ..utils import get_src_loc
from .base import Primitive, register_primitive


@register_primitive()
class SplitPrimitive(Primitive):
    name = "split"
    is_stage_primitive = True

    @staticmethod
    def apply(stage, parent, factor=None, nparts=None, mode="transform"):
        """Split the stage either by factor providing outer scope, or both"""
        from ..schedule import Schedule

        sch = Schedule._CurrentSchedule
        if sch.is_lowered():
            raise APIError(".split() must be called before lowering")
        if nparts is not None or mode != "transform":
            raise HCLNotImplementedError(f"nparts={nparts}, mode={mode} not supported")
        if isinstance(parent, int):
            parent = stage.tensor.axis[parent]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        split_op = ast.SplitOp(stage.stage_handle, parent, factor, loc)
        sch.ast.top_func.body.append(split_op)
        return split_op.results[0], split_op.results[1]
