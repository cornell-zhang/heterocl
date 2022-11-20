# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2022 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #

from .ir import intermediate as itmd
from .types import *

""" Type inference rules """
def add_sub_rule():
    ops = (itmd.Add, itmd.Sub, itmd.SelectOp)
    int_rules = {
        (Int, Int) : lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (Int, UInt): lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (UInt, UInt): lambda t1, t2: UInt(max(t1.bits, t2.bits)),
        (Index, Index): lambda t1, t2: Index()
    }
    float_rules = {
        (Float, Float) : lambda t1, t2: Float(max(t1.bits, t2.bits)),
        (Float, Int) : lambda t1, t2 : t1 if isinstance(t1, Float) else t2,
        (Float, UInt) : lambda t1, t2 : t1 if isinstance(t1, Float) else t2,
    }
    #TODO: commutative rule as an input to TypeRule
    return TypeRule(ops, [int_rules, float_rules])

def cmp_rule():
    ops = (itmd.Cmp,)
    int_rules = {
        (Int, Int) : lambda t1, t2: (Int(max(t1.bits, t2.bits)), UInt(1)),
        (Int, UInt): lambda t1, t2: (Int(max(t1.bits, t2.bits)), UInt(1)),
        (UInt, UInt): lambda t1, t2: (UInt(max(t1.bits, t2.bits)), UInt(1)),
        (Index, Index): lambda t1, t2: (Index(), UInt(1))
    }
    float_rules = {
        (Float, Float) : lambda t1, t2: (Float(max(t1.bits, t2.bits)), UInt(1)),
        (Float, Int) : lambda t1, t2 : (t1 if isinstance(t1, Float) else t2, UInt(1)),
        (Float, UInt) : lambda t1, t2 : (t1 if isinstance(t1, Float) else t2, UInt(1)),
    }
    return TypeRule(ops, [int_rules, float_rules])


def and_or_rule():
    ops = (itmd.And, itmd.Or)
    int_rules = {
        (Int, Int) : lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (Int, UInt): lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (UInt, UInt): lambda t1, t2: UInt(max(t1.bits, t2.bits)),
    }
    return TypeRule(ops, [int_rules])



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

def logic_op_rule():
    ops = (itmd.LogicalAnd, itmd.LogicalOr, itmd.LogicalXOr)
    int_rules = {
        (Int, Int) : lambda t1, t2: UInt(1),
        (Int, UInt): lambda t1, t2: UInt(1),
        (UInt, UInt): lambda t1, t2: UInt(1),
    }
    return TypeRule(ops, [int_rules])

#TODO: attach typing rules to itmd.Operation classes
# Make this a hook? more extensible
def get_type_rules():
    rules = list()
    rules.append(add_sub_rule())
    rules.append(mul_rule())
    rules.append(mod_rule())
    rules.append(and_or_rule())
    rules.append(logic_op_rule())
    rules.append(cmp_rule())
    return rules