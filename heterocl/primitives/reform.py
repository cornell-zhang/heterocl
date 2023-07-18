# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import (
    APIError,
)
from ..ast import ast
from ..utils import get_src_loc
from .base import Primitive, register_primitive


@register_primitive()
class ReformPrimitive(Primitive):
    name = "reform"

    @staticmethod
    def apply(sch, target, layout):
        """Change the layout of a tensor"""
        if sch.is_lowered():
            raise APIError(".reform() must be called before lowering")
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        reform_op = ast.ReformOp(target, layout, loc)
        sch.ast.top_func.body.append(reform_op)
