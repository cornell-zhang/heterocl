# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, arguments-differ

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


@register_primitive()
class ReorderPrimitive(Primitive):
    name = "reorder"
    is_stage_primitive = True

    @staticmethod
    def apply(stage, *args):
        """reorder the arguments in the specified order."""
        from ..schedule import Schedule

        sch = Schedule._CurrentSchedule
        if sch.is_lowered():
            raise APIError(".reorder() must be called before lowering")
        args = list(args)
        # pylint: disable=consider-using-enumerate
        for i in range(len(args)):
            if isinstance(args[i], int):
                args[i] = stage.tensor.axis[args[i]]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        reorder_op = ast.ReorderOp(args, loc)
        sch.ast.top_func.body.append(reorder_op)
        op = reorder_op
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        ir_builder = IRBuilder(sch._ast)
        ip = InsertionPoint.at_block_terminator(sch.top_func.entry_block)
        for arg in op.args:
            ir_builder.build_visitor(arg, ip)
        arg_results = [arg.result for arg in op.args]
        hcl_reorder_op = hcl_d.ReorderOp(arg_results, ip=ip, loc=loc)
        op.ir_op = hcl_reorder_op
