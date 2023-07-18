# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import (
    APIError,
)

from ..ast import ast
from ..utils import get_src_loc
from .base import Primitive, register_primitive


@register_primitive()
class ReplacePrimitive(Primitive):
    name = "replace"

    @staticmethod
    def apply(sch, src, dst):
        """Replace a Tensor with another Tensor"""
        if sch.is_lowered():
            raise APIError(".replace() must be called before lowering")

        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        replace_op = ast.ReplaceOp(src, dst, loc)
        sch.ast.top_func.body.append(replace_op)
