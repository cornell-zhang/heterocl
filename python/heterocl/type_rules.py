# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2022 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #

from .ir import intermediate as itmd
from .types import *

""" Type inference rules """
def add_sub_rule():
    ops = (itmd.Add, itmd.Sub, itmd.Cmp)
    int_rules = {
        (Int, Int) : lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (Int, UInt): lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (Index, Index): lambda t1, t2: Index()
    }
    float_rules = {
        (Float, Float) : lambda t1, t2: Float(max(t1.bits, t2.bits)),
    }
    return TypeRule(ops, [int_rules, float_rules])


def mul_rule():
    ops = (itmd.Mul)
    int_rules = {
        (Int, Int) : lambda t1, t2: Int(t1.bits * t2.bits),
        (Int, UInt): lambda t1, t2: Int(t1.bits * t2.bits)
    }
    float_rules = {
        (Float, Float) : lambda t1, t2: Float(max(t1.bits, t2.bits)),
    }
    return TypeRule(ops, [int_rules, float_rules])


def mod_rule():
    ops = (itmd.Mod)
    int_rules = {
        (Int, Int) : lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (Int, UInt): lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (Index, Index): lambda t1, t2: Index()
    }
    return TypeRule(ops, [int_rules])

def get_type_rules():
    rules = list()
    rules.append(add_sub_rule())
    rules.append(mul_rule())
    rules.append(mod_rule())
    return rules