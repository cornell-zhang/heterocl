# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, arguments-differ, unused-argument

from hcl_mlir.exceptions import (
    APIError,
)
from hcl_mlir.ir import (
    Location,
    InsertionPoint,
    F32Type,
    MemRefType,
)
from hcl_mlir.dialects import hcl as hcl_d
from ..ast import ast
from ..ast.ir_builder import IRBuilder
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
        op = buffer_at_op
        ir_builder = IRBuilder(sch._ast)
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        ip = InsertionPoint.at_block_terminator(sch.top_func.entry_block)
        ir_builder.build_visitor(op.target, ip)
        ir_builder.build_visitor(op.axis, ip)
        f32 = F32Type.get()
        memref_type = MemRefType.get((1,), f32, loc=loc)
        hcl_buffer_at_op = hcl_d.BufferAtOp(
            memref_type, op.target.result, op.axis.result, ip=ip, loc=loc
        )
        op.ir_op = hcl_buffer_at_op
        op.result = hcl_buffer_at_op.result
        return buffer_at_op
