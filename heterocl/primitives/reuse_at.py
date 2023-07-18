# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import (
    APIError,
    DTypeError,
)
from ..ast import ast
from ..utils import get_src_loc
from .base import Primitive, register_primitive


@register_primitive()
class ReuseAtPrimitive(Primitive):
    name = "reuse_at"

    @staticmethod
    def apply(sch, target, parent, axis, name=None):
        if sch.is_lowered():
            raise APIError(".reuse_at() must be called before lowering")
        if not isinstance(axis, ast.LoopHandle):
            raise DTypeError(f"reuse_at() got invalid axis of type {type(axis)}")
        if not isinstance(target, (ast.AllocOp, ast.ReuseAtOp)):
            raise DTypeError(f"reuse_at() got invalid target of type {type(target)}")

        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        reuse_at_op = ast.ReuseAtOp(target, axis, loc)
        sch.ast.top_func.body.append(reuse_at_op)
        return reuse_at_op
