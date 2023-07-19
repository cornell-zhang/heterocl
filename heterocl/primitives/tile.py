# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, arguments-differ

from hcl_mlir.exceptions import (
    APIError,
)
from hcl_mlir.ir import (
    Location,
    InsertionPoint,
    IntegerType,
    IntegerAttr,
)
from hcl_mlir.dialects import hcl as hcl_d
from ..ast import ast
from ..ast.ir_builder import IRBuilder
from ..utils import get_src_loc
from .base import Primitive, register_primitive


@register_primitive()
class TilePrimitive(Primitive):
    name = "tile"
    is_stage_primitive = True

    @staticmethod
    def apply(stage, x_parent, y_parent, x_factor, y_factor):
        """Perform tiling on two dimensions"""
        from ..schedule import Schedule

        sch = Schedule._CurrentSchedule
        if sch.is_lowered():
            raise APIError(".tile() must be called before lowering")
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        tile_op = ast.TileOp(
            stage.stage_handle, x_parent, y_parent, x_factor, y_factor, loc
        )
        sch.ast.top_func.body.append(tile_op)
        op = tile_op
        res0, res1, res2, res3 = (
            tile_op.results[0],
            tile_op.results[1],
            tile_op.results[2],
            tile_op.results[3],
        )
        ir_builder = IRBuilder(sch._ast)
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        ip = InsertionPoint.at_block_terminator(sch.top_func.entry_block)
        i32 = IntegerType.get_unsigned(32)
        x_factor = IntegerAttr.get(i32, op.x_factor)
        y_factor = IntegerAttr.get(i32, op.y_factor)
        ir_builder.build_visitor(op.x_parent, ip)
        ir_builder.build_visitor(op.y_parent, ip)
        hcl_tile_op = hcl_d.TileOp(
            op.x_parent.result,
            op.y_parent.result,
            x_factor,
            y_factor,
            ip=ip,
            loc=loc,
        )
        op.ir_op = hcl_tile_op
        for result_loop_hdl, hdl_result in zip(op.results, hcl_tile_op.results):
            result_loop_hdl.result = hdl_result
        return (res0, res1, res2, res3)
