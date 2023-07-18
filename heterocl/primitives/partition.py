# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from hcl_mlir.exceptions import (
    HCLValueError,
    APIError,
)

from ..ast import ast
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
