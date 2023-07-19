# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, arguments-differ

import functools

from hcl_mlir.exceptions import (
    APIError,
)
from hcl_mlir.ir import (
    Location,
    InsertionPoint,
    MemRefType,
)
from hcl_mlir.dialects import hcl as hcl_d
from ..ast import ast
from ..ast.ir_builder import IRBuilder
from ..utils import get_src_loc, hcl_dtype_to_mlir
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
        op = reshape_op
        ir_builder = IRBuilder(sch._ast)
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        # Be careful, since arg.result has been removed when building func op
        if op.tensor.result is None:
            op.tensor.result = op.tensor.prev_result
        ip = InsertionPoint.at_block_terminator(sch.top_func.entry_block)
        ir_builder.build_visitor(op.tensor, ip)
        eletype = hcl_dtype_to_mlir(op.tensor.dtype)
        memref_type = MemRefType.get(op.shape, eletype, loc=loc)
        hcl_reshape_op = hcl_d.ReshapeOp(memref_type, op.tensor.result, ip=ip, loc=loc)
        op.ir_op = hcl_reshape_op
