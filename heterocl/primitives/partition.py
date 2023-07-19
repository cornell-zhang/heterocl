# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, arguments-differ

from hcl_mlir.exceptions import (
    HCLValueError,
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


class Partition:
    Complete = 0
    Block = 1
    Cyclic = 2


@register_primitive()
class ParitionPrimitive(Primitive):
    name = "partition"

    @staticmethod
    def apply(sch, target, partition_type=Partition.Complete, dim=0, factor=0):
        """Partition a Tensor into smaller Tensors or even registers"""
        if sch.is_lowered():
            raise APIError(".partition() must be called before lowering")
        if partition_type > 2:
            raise HCLValueError("Invalid partition type")
        if dim < 0:
            raise HCLValueError("Invalid dimension")
        if factor < 0:
            raise HCLValueError("Invalid factor")

        filename, lineno = get_src_loc()
        loc = ast.Location(filename, lineno)
        if partition_type == Partition.Complete:
            partition_type = 0
        elif partition_type == Partition.Block:
            partition_type = 1
        elif partition_type == Partition.Cyclic:
            partition_type = 2
        else:
            raise HCLValueError("Not supported partition type")
        partition_op = ast.PartitionOp(target, partition_type, dim, factor, loc)
        sch.ast.top_func.body.append(partition_op)
        op = partition_op
        i32 = IntegerType.get_signless(32)
        ui32 = IntegerType.get_unsigned(32)
        partition_type = IntegerAttr.get(i32, op.kind)
        dim = IntegerAttr.get(ui32, op.dim)
        factor = IntegerAttr.get(ui32, op.factor)
        ir_builder = IRBuilder(sch._ast)
        loc = Location.file(op.loc.filename, op.loc.lineno, 0)
        # Be careful, since arg.result has been removed when building func op
        if op.tensor.result is None:
            op.tensor.result = op.tensor.prev_result
        ip = InsertionPoint.at_block_terminator(sch.top_func.entry_block)
        ir_builder.build_visitor(op.tensor, ip)
        hcl_partition_op = hcl_d.PartitionOp(
            op.tensor.result,
            partition_kind=partition_type,
            dim=dim,
            factor=factor,
            ip=ip,
            loc=loc,
        )
        op.ir_op = hcl_partition_op
