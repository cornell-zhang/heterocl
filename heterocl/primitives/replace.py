# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import (
    APIError,
)
from hcl_mlir.ir import (
    Location,
    InsertionPoint,
)
from hcl_mlir.dialects import hcl as hcl_d
from ..ast import ast
from ..ast.ir_builder import IRBuilder
from ..utils import get_src_loc
from .base import Primitive, register_primitive
from ..context import get_context, get_location


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
        op = replace_op
        with get_context(), get_location():
            loc = Location.file(op.loc.filename, op.loc.lineno, 0)
            ir_builder = IRBuilder(sch._ast)
            ip = InsertionPoint.at_block_terminator(sch.top_func.entry_block)
            ir_builder.build_visitor(op.target, ip)
            ir_builder.build_visitor(op.src, ip)
            replace_op = hcl_d.ReplaceOp(
                op.target.result, op.src.result, ip=ip, loc=loc
            )
            op.ir_op = replace_op
