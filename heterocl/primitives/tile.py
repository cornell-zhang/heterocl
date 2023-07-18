# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import (
    APIError,
)

from ..ast import ast
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
        return (
            tile_op.results[0],
            tile_op.results[1],
            tile_op.results[2],
            tile_op.results[3],
        )
