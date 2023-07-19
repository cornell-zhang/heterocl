# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, arguments-differ, unused-argument

from hcl_mlir.exceptions import (
    APIError,
    DTypeError,
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
        ast_reuse_at_op = ast.ReuseAtOp(target, axis, loc)
        sch.ast.top_func.body.append(ast_reuse_at_op)
        op = ast_reuse_at_op
        ir_builder = IRBuilder(sch._ast)
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        # Be careful, since arg.result has been removed when building func op
        if op.target.result is None:
            op.target.result = op.target.prev_result
        ip = InsertionPoint.at_block_terminator(sch.top_func.entry_block)
        ir_builder.build_visitor(op.target, ip)
        ir_builder.build_visitor(op.axis, ip)
        f32 = F32Type.get()
        memref_type = MemRefType.get((1,), f32, loc=loc)
        reuse_at_op = hcl_d.ReuseAtOp(
            memref_type, op.target.result, op.axis.result, ip=ip, loc=loc
        )
        op.ir_op = reuse_at_op
        op.result = reuse_at_op.result
        return ast_reuse_at_op
