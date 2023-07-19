# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, arguments-differ

from hcl_mlir.exceptions import (
    APIError,
    HCLNotImplementedError,
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
class SplitPrimitive(Primitive):
    name = "split"
    is_stage_primitive = True

    @staticmethod
    def apply(stage, parent, factor=None, nparts=None, mode="transform"):
        """Split the stage either by factor providing outer scope, or both"""
        from ..schedule import Schedule

        sch = Schedule._CurrentSchedule
        if sch.is_lowered():
            raise APIError(".split() must be called before lowering")
        if nparts is not None or mode != "transform":
            raise HCLNotImplementedError(f"nparts={nparts}, mode={mode} not supported")
        if isinstance(parent, int):
            parent = stage.tensor.axis[parent]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        split_op = ast.SplitOp(stage.stage_handle, parent, factor, loc)
        sch.ast.top_func.body.append(split_op)
        op = split_op
        res0, res1 = split_op.results[0], split_op.results[1]
        ir_builder = IRBuilder(sch._ast)
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        ip = InsertionPoint.at_block_terminator(sch.top_func.entry_block)
        ir_builder.build_visitor(op.parent, ip)
        i32 = IntegerType.get_unsigned(32)
        factor = IntegerAttr.get(i32, op.factor)
        hcl_split_op = hcl_d.SplitOp(op.parent.result, factor, ip=ip, loc=loc)
        op.ir_op = hcl_split_op
        for result_loop_hdl, hdl_result in zip(op.results, hcl_split_op.results):
            result_loop_hdl.result = hdl_result
        return res0, res1
