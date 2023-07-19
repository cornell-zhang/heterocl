# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import hcl_mlir
from hcl_mlir import UnitAttr


def get_affine_loop_nests(func):
    loops = hcl_mlir.get_affine_loop_nests(func)[0]
    res = []
    for item in loops:
        res.append((item["name"], item["body"]))
    return res


def annotate(op, name):
    op.attributes[name] = UnitAttr.get()
