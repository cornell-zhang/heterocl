# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2022 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #

from .ir import intermediate as itmd
from .types import *

""" Type inference rules """
def add_sub_rule():
    ops = (itmd.AddOp, itmd.SubOp)
    int_rules = {
        (Int, Int) : lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (Int, UInt): lambda t1, t2: Int(max(t1.bits, t2.bits))
    }
    float_rules = {
        (Float, Float) : lambda t1, t2: Float(max(t1.bits, t2.bits)),
    }
    # TODO: merge the rule dicts in TypeRule
    return TypeRule(ops, [int_rules, float_rules])