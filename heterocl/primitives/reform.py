# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, arguments-differ

from hcl_mlir.exceptions import (
    APIError,
)
from hcl_mlir.ir import (
    Location,
    InsertionPoint,
    AffineMap,
    AffineMapAttr,
    MemRefType,
)
from hcl_mlir.dialects import hcl as hcl_d
from ..ast import ast
from ..ast.ir_builder import IRBuilder
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
        op = reform_op
        ir_builder = IRBuilder(sch._ast)
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        ip = InsertionPoint.at_block_terminator(sch.top_func.entry_block)
        ir_builder.build_visitor(op.target, ip)
        if op.layout == "nhwc":
            attr = AffineMap.get_permutation([0, 2, 3, 1])
        else:
            raise RuntimeError("Not supported layout")
        memref_type = MemRefType.get(op.target.shape, op.target.ir_op.dtype)
        hcl_reform_op = hcl_d.ReformOp(memref_type, op.target.result, ip=ip, loc=loc)
        hcl_reform_op.attributes["layout"] = AffineMapAttr.get(attr)
        op.ir_op = hcl_reform_op
