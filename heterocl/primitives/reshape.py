# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import functools

from hcl_mlir.exceptions import (
    APIError,
)
from ..ast import ast
from ..utils import get_src_loc
from .base import Primitive, register_primitive


@register_primitive()
class ReshapePrimitive(Primitive):
    name = "reshape"

    @staticmethod
    def apply(sch, target, shape):
        """Reshape a Tensor to a specified new shape"""
        if sch.is_lowered():
            raise APIError(".reshape() must be called before lowering")
        ori_size = functools.reduce(lambda a, b: a * b, target.shape, 1)
        new_size = functools.reduce(lambda a, b: a * b, shape, 1)
        if ori_size != new_size:
            raise RuntimeError(
                "The reshaped tensor should have the same total size with the original tensor"
            )
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        reshape_op = ast.ReshapeOp(target, shape, loc)
        sch.ast.top_func.body.append(reshape_op)
