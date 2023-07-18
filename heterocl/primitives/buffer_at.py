# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import (
    APIError,
)
from ..ast import ast
from ..utils import get_src_loc
from .base import Primitive, register_primitive


@register_primitive()
class BufferAtPrimitive(Primitive):
    name = "buffer_at"

    @staticmethod
    def apply(sch, target, parent, axis, name=None):
        """Create a write buffer reusing the output of current stage"""
        if sch.is_lowered():
            raise APIError(".buffer_at() must be called before lowering")

        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        buffer_at_op = ast.BufferAtOp(target, axis, loc)
        sch.ast.top_func.body.append(buffer_at_op)
        return buffer_at_op
