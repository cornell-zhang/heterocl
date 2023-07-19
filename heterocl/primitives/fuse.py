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
class FusePrimitive(Primitive):
    name = "fuse"
    is_stage_primitive = True

    @staticmethod
    def apply(stage, *args):
        """Fuse multiple consecutive iteration variables into a single iteration variable."""
        from ..schedule import Schedule

        sch = Schedule._CurrentSchedule
        if sch.is_lowered():
            raise APIError(".fuse() must be called before lowering")
        assert len(args) >= 1, "Length of the arguments must be >=1 for fuse."
        args = list(args)
        # pylint: disable=consider-using-enumerate
        for i in range(len(args)):
            if isinstance(args[i], int):
                args[i] = stage.tensor.axis[args[i]]
        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        fuse_op = ast.FuseOp(args, loc)
        sch.ast.top_func.body.append(fuse_op)
        op = fuse_op
        ir_builder = IRBuilder(sch._ast)
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        ip = InsertionPoint.at_block_terminator(sch.top_func.entry_block)
        for arg in op.arg_list:
            ir_builder.build_visitor(arg, ip)
        arg_results = [arg.result for arg in op.arg_list]
        hcl_fuse_op = hcl_d.FuseOp(arg_results, ip=ip, loc=loc)
        op.ir_op = hcl_fuse_op
        op.result = hcl_fuse_op.result
        return fuse_op
