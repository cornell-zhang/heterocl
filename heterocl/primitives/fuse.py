# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import (
    APIError,
)

from ..ast import ast
from ..utils import get_src_loc
from .base import Primitive, register_primitive


@register_primitive()
class FusePrimitive(Primitive):
    name = "fuse"
    is_stage_primitive = True

    @staticmethod
    def apply(stage, *args):
        """Fuse multiple consecutive iteration variables into a single iteration variable."""
        from ..schedule import Schedule

        sch = Schedule._CurrentSchedule
        if sch.is_lowered():
            raise APIError(".fuse() must be called before lowering")
        assert len(args) >= 1, "Length of the arguments must be >=1 for fuse."
        args = list(args)
        # pylint: disable=consider-using-enumerate
        for i in range(len(args)):
            if isinstance(args[i], int):
                args[i] = stage.tensor.axis[args[i]]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        fuse_op = ast.FuseOp(args, loc)
        sch.ast.top_func.body.append(fuse_op)
        return fuse_op
