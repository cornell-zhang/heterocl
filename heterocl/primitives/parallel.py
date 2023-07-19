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
class ParallelPrimitive(Primitive):
    name = "parallel"
    is_stage_primitive = True

    @staticmethod
    def apply(stage, var):
        """Parallelize the iteration."""
        from ..schedule import Schedule

        sch = Schedule._CurrentSchedule
        if sch.is_lowered():
            raise APIError(".parallel() must be called before lowering")
        if isinstance(var, int):
            var = stage.tensor.axis[var]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        parallel_op = ast.ParallelOp(var, loc)
        sch.ast.top_func.body.append(parallel_op)
        op = parallel_op
        ir_builder = IRBuilder(sch._ast)
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        ip = InsertionPoint.at_block_terminator(sch.top_func.entry_block)
        ir_builder.build_visitor(op.target, ip)
        hcl_parallel_op = hcl_d.ParallelOp(op.target.result, ip=ip, loc=loc)
        op.ir_op = hcl_parallel_op
