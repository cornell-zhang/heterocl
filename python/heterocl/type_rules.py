# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2022 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #

from .ir import intermediate as itmd
from .types import *

""" Type inference rules """
def add_sub_rule():
    ops = (itmd.Add, itmd.Sub)
    int_rules = {
        (Int, Int) : lambda t1, t2: Int(max(t1.bits, t2.bits) + 1),
        (Int, UInt): lambda t1, t2: Int(max(t1.bits, t2.bits) + 1)
    }
    float_rules = {
        (Float, Float) : lambda t1, t2: Float(max(t1.bits, t2.bits)),
    }
    return TypeRule(ops, [int_rules, float_rules])