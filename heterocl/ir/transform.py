# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module, no-value-for-parameter

import hcl_mlir
from hcl_mlir import UnitAttr, StringAttr, InsertionPoint, MemRefType
from hcl_mlir.dialects import memref as memref_d


def get_affine_loop_nests(func):
    loops = hcl_mlir.get_affine_loop_nests(func)
    res = []
    for loop in loops:
        res.append([(item["name"], item["body"]) for item in loop])
    return res


def annotate(op, name):
    op.attributes[name] = UnitAttr.get()


def create_buffer(tensor, name, ip):
    with InsertionPoint(ip):
        alloc_op = memref_d.AllocOp(tensor.type, [], [])
        alloc_op.attributes["name"] = StringAttr.get(name)
        shape = MemRefType(tensor.type).shape
        for_loops = []

        def recursive_for(for_handle, idx):
            if idx == len(shape):
                return
            with InsertionPoint(for_handle.body.operations[0]):
                new_for = hcl_mlir.make_for(
                    0, shape[idx], name=name + f"_l_{idx}", stage="S_" + name
                )
                for_loops.append(new_for)
                recursive_for(new_for, idx + 1)

        for_handle = hcl_mlir.make_for(
            0, shape[0], name=name + "_l_0", stage="S_" + name
        )
        for_loops.append(for_handle)
        recursive_for(for_handle, 1)
        induction_vars = [for_loop.induction_variable for for_loop in for_loops]
        with InsertionPoint(for_loops[-1].body.operations[0]):
            load = memref_d.LoadOp(tensor, induction_vars)
            memref_d.StoreOp(
                load.result,
                alloc_op.result,
                induction_vars,
            )
        # TODO: Upgrade LLVM version and use the following code
        # tensor.replace_all_uses_with(alloc_op.result)
