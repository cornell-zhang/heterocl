# ===----------------------------------------------------------------------=== #
#
# Copyright 2021-2023 The HCL-MLIR Authors.
#
# ===----------------------------------------------------------------------=== #

from .ast import ast
from .types import *

# TODO: Reduction op rules

""" Type inference rules """


def add_sub_rule():
    ops = (ast.Add, ast.Sub)
    int_rules = {
        (Int, Int): lambda t1, t2: Int(max(t1.bits, t2.bits) + 1),
        (Int, UInt): lambda t1, t2: Int(max(t1.bits, t2.bits + 1) + 1),
        (Int, Index): lambda t1, t2: Int(max(t1.bits, t2.bits + 1) + 1),
        (Int, Fixed): lambda t1, t2: Fixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs + 1, t2.fracs
        ),
        (Int, UFixed): lambda t1, t2: Fixed(
            max(t1.bits, t2.bits - t2.fracs + 1) + t2.fracs + 1, t2.fracs
        ),
        (Int, Float): lambda t1, t2: t2,
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: Int(max(t1.bits + 1, t2.bits) + 1),
        (UInt, UInt): lambda t1, t2: UInt(max(t1.bits, t2.bits) + 1),
        (UInt, Index): lambda t1, t2: UInt(max(t1.bits, t2.bits) + 1),
        (UInt, Fixed): lambda t1, t2: Fixed(
            max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs + 1, t2.fracs
        ),
        (UInt, UFixed): lambda t1, t2: UFixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs + 1, t2.fracs
        ),
        (UInt, Float): lambda t1, t2: t2,
    }
    index_rules = {
        (Index, Int): lambda t1, t2: Int(max(t1.bits + 1, t2.bits) + 1),
        (Index, UInt): lambda t1, t2: UInt(max(t1.bits, t2.bits) + 1),
        (Index, Index): lambda t1, t2: Index(),
        (Index, Fixed): lambda t1, t2: Fixed(
            max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs + 1, t2.fracs
        ),
        (Index, UFixed): lambda t1, t2: UFixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs + 1, t2.fracs
        ),
        (Index, Float): lambda t1, t2: t2,
    }
    fixed_rules = {
        (Fixed, Int): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits) + t1.fracs + 1, t1.fracs
        ),
        (Fixed, UInt): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs + 1, t1.fracs
        ),
        (Fixed, Index): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs + 1, t1.fracs
        ),
        (Fixed, Fixed): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs) + 1,
            max(t1.fracs, t2.fracs),
        ),
        (Fixed, UFixed): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs + 1)
            + max(t1.fracs, t2.fracs)
            + 1,
            max(t1.fracs, t2.fracs),
        ),
        (Fixed, Float): lambda t1, t2: t2,
    }
    ufixed_rules = {
        (UFixed, Int): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs + 1, t2.bits) + t1.fracs + 1, t1.fracs
        ),
        (UFixed, UInt): lambda t1, t2: UFixed(
            max(t1.bits - t1.fracs, t2.bits) + t1.fracs + 1, t1.fracs
        ),
        (UFixed, Index): lambda t1, t2: UFixed(
            max(t1.bits - t1.fracs, t2.bits) + t1.fracs + 1, t1.fracs
        ),
        (UFixed, Fixed): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs + 1, t2.bits - t2.fracs)
            + max(t1.fracs, t2.fracs)
            + 1,
            max(t1.fracs, t2.fracs),
        ),
        (UFixed, UFixed): lambda t1, t2: UFixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs) + 1,
            max(t1.fracs, t2.fracs),
        ),
        (UFixed, Float): lambda t1, t2: t2,
    }
    float_rules = {
        (Float, Int): lambda t1, t2: t1,
        (Float, UInt): lambda t1, t2: t1,
        (Float, Index): lambda t1, t2: t1,
        (Float, Fixed): lambda t1, t2: t1,
        (Float, UFixed): lambda t1, t2: t1,
        (Float, Float): lambda t1, t2: Float(max(t1.bits, t2.bits)),
    }
    return TypeRule(
        ops,
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
    )


def mul_rule():
    ops = ast.Mul
    int_rules = {
        (Int, Int): lambda t1, t2: Int(t1.bits + t2.bits),
        (Int, UInt): lambda t1, t2: Int(t1.bits + t2.bits),
        (Int, Index): lambda t1, t2: Int(t1.bits + t2.bits),
        (Int, Fixed): lambda t1, t2: Fixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        (Int, UFixed): lambda t1, t2: Fixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        (Int, Float): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
    }
    uint_rules = {
        # (Uint, Int) covered by (Int, Uint)
        (UInt, UInt): lambda t1, t2: UInt(t1.bits + t2.bits),
        (UInt, Index): lambda t1, t2: UInt(t1.bits + t2.bits),
        (UInt, Fixed): lambda t1, t2: Fixed(t1.bits + t2.bits, max(t1.fracs, t2.fracs)),
        (UInt, UFixed): lambda t1, t2: UFixed(
            t1.bits + t2.bits, max(t1.fracs, t2.fracs)
        ),
        (UInt, Float): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
    }
    index_rules = {
        # (Index, Int) covered by (Int, Index)
        # (Index, UInt) covered by (UInt, Index)
        (Index, Index): lambda t1, t2: Index(),
        (Index, Fixed): lambda t1, t2: Fixed(
            t1.bits + t2.bits, max(t1.fracs, t2.fracs)
        ),
        (Index, UFixed): lambda t1, t2: UFixed(
            t1.bits + t2.bits, max(t1.fracs, t2.fracs)
        ),
        (Index, Float): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
    }
    fixed_rules = {
        # (Fixed, Int) covered by (Int, Fixed)
        # (Fixed, UInt) covered by (UInt, Fixed)
        # (Fixed, Index) covered by (Index, Fixed)
        (Fixed, Fixed): lambda t1, t2: Fixed(t1.bits + t2.bits, t1.fracs + t2.fracs),
        (Fixed, UFixed): lambda t1, t2: Fixed(t1.bits + t2.bits, t1.fracs + t2.fracs),
        (Fixed, Float): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
    }
    ufixed_rules = {
        # (UFixed, Int) covered by (Int, UFixed)
        # (UFixed, UInt) covered by (UInt, UFixed)
        # (UFixed, Index) covered by (Index, UFixed)
        # (UFixed, Fixed) covered by (Fixed, UFixed)
        (UFixed, UFixed): lambda t1, t2: UFixed(t1.bits + t2.bits, t1.fracs + t2.fracs),
        (UFixed, Float): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
    }
    float_rules = {
        # (Float, (Int, UInt, Index, Fixed, UFixed)) covered
        (Float, Float): lambda t1, t2: Float(max(t1.bits, t2.bits))
    }
    return TypeRule(
        ops,
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
        commutative=True,
    )


def div_rule():
    ops = (ast.Div, ast.FloorDiv)
    int_rules = {
        (Int, Int): lambda t1, t2: t1,
        (Int, UInt): lambda t1, t2: t1,
        (Int, Index): lambda t1, t2: t1,
        (Int, Fixed): lambda t1, t2: Fixed(t1.bits + t2.bits, t1.bits - t2.fracs),
        (Int, UFixed): lambda t1, t2: Fixed(t1.bits + t2.bits + 1, t1.bits - t2.fracs),
        (Int, Float): lambda t1, t2: t2,
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: Int(t1.bits),
        (UInt, UInt): lambda t1, t2: t1,
        (UInt, Index): lambda t1, t2: t1,
        (UInt, Fixed): lambda t1, t2: Fixed(t1.bits + t2.bits, t1.bits - t2.fracs),
        (UInt, UFixed): lambda t1, t2: UFixed(t1.bits + t2.bits, t1.bits - t2.fracs),
        (UInt, Float): lambda t1, t2: t2,
    }
    index_rules = {
        (Index, Int): lambda t1, t2: Int(t1.bits),
        (Index, UInt): lambda t1, t2: t1,
        (Index, Index): lambda t1, t2: Index(),
        (Index, Fixed): lambda t1, t2: Fixed(t1.bits + t2.bits, t1.bits - t2.fracs),
        (Index, UFixed): lambda t1, t2: UFixed(t1.bits + t2.bits, t1.bits - t2.fracs),
        (Index, Float): lambda t1, t2: t2,
    }
    fixed_rules = {
        (Fixed, Int): lambda t1, t2: Fixed(t1.bits + t2.bits, t2.bits + t1.fracs),
        (Fixed, UInt): lambda t1, t2: Fixed(t1.bits + t2.bits + 1, t2.bits + t1.fracs),
        (Fixed, Index): lambda t1, t2: Fixed(t1.bits + t2.bits + 1, t2.bits + t1.fracs),
        (Fixed, Fixed): lambda t1, t2: Fixed(
            t1.bits + t2.bits, t2.bits - t2.fracs + t1.fracs
        ),
        (Fixed, UFixed): lambda t1, t2: Fixed(
            t1.bits + t2.bits + 1, t2.bits - t2.fracs + t1.fracs
        ),
        (Fixed, Float): lambda t1, t2: t2,
    }
    ufixed_rules = {
        (UFixed, Int): lambda t1, t2: Fixed(t1.bits + t2.bits + 1, t2.bits + t1.fracs),
        (UFixed, UInt): lambda t1, t2: UFixed(t1.bits + t2.bits, t2.bits + t1.fracs),
        (UFixed, Index): lambda t1, t2: UFixed(t1.bits + t2.bits, t2.bits + t1.fracs),
        (UFixed, Fixed): lambda t1, t2: Fixed(
            t1.bits + t2.bits, t2.bits - t2.fracs + t1.fracs
        ),
        (UFixed, UFixed): lambda t1, t2: UFixed(
            t1.bits + t2.bits, t2.bits - t2.fracs + t1.fracs
        ),
        (UFixed, Float): lambda t1, t2: t2,
    }
    float_rules = {
        (Float, Int): lambda t1, t2: t1,
        (Float, UInt): lambda t1, t2: t1,
        (Float, Index): lambda t1, t2: t1,
        (Float, Fixed): lambda t1, t2: t1,
        (Float, UFixed): lambda t1, t2: t1,
        (Float, Float): lambda t1, t2: Float(max(t1.bits, t2.bits)),
    }
    return TypeRule(
        ops,
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
    )


def mod_rule():
    ops = ast.Mod
    int_rules = {
        (Int, Int): lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (Int, UInt): lambda t1, t2: Int(max(t1.bits, t2.bits + 1)),
        (Int, Index): lambda t1, t2: Int(max(t1.bits, t2.bits + 1)),
        (Int, Fixed): lambda t1, t2: Fixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (Int, UFixed): lambda t1, t2: Fixed(
            max(t1.bits, t2.bits - t2.fracs + 1) + t2.fracs, t2.fracs
        ),
        (Int, Float): lambda t1, t2: t2,
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: Int(max(t1.bits + 1, t2.bits)),
        (UInt, UInt): lambda t1, t2: UInt(max(t1.bits, t2.bits)),
        (UInt, Index): lambda t1, t2: UInt(max(t1.bits, t2.bits)),
        (UInt, Fixed): lambda t1, t2: Fixed(
            max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (UInt, UFixed): lambda t1, t2: UFixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (UInt, Float): lambda t1, t2: t2,
    }
    index_rules = {
        (Index, Int): lambda t1, t2: Int(max(t1.bits + 1, t2.bits)),
        (Index, UInt): lambda t1, t2: UInt(max(t1.bits, t2.bits)),
        (Index, Index): lambda t1, t2: Index(),
        (Index, Fixed): lambda t1, t2: Fixed(
            max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (Index, UFixed): lambda t1, t2: UFixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (Index, Float): lambda t1, t2: t2,
    }
    fixed_rules = {
        (Fixed, Int): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs
        ),
        (Fixed, UInt): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs, t1.fracs
        ),
        (Fixed, Index): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs, t1.fracs
        ),
        (Fixed, Fixed): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
            max(t1.fracs, t2.fracs),
        ),
        (Fixed, UFixed): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs + 1) + max(t1.fracs, t2.fracs),
            max(t1.fracs, t2.fracs),
        ),
        (Fixed, Float): lambda t1, t2: t2,
    }
    ufixed_rules = {
        (UFixed, Int): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs + 1, t2.bits) + t1.fracs, t1.fracs
        ),
        (UFixed, UInt): lambda t1, t2: UFixed(
            max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs
        ),
        (UFixed, Index): lambda t1, t2: UFixed(
            max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs
        ),
        (UFixed, Fixed): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs + 1, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
            max(t1.fracs, t2.fracs),
        ),
        (UFixed, UFixed): lambda t1, t2: UFixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
            max(t1.fracs, t2.fracs),
        ),
        (UFixed, Float): lambda t1, t2: t2,
    }
    float_rules = {
        (Float, Int): lambda t1, t2: t1,
        (Float, UInt): lambda t1, t2: t1,
        (Float, Index): lambda t1, t2: t1,
        (Float, Fixed): lambda t1, t2: t1,
        (Float, UFixed): lambda t1, t2: t1,
        (Float, Float): lambda t1, t2: Float(max(t1.bits, t2.bits)),
    }
    return TypeRule(
        ops,
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
    )


def cmp_rule():
    ops = (ast.Cmp,)
    int_rules = {
        (Int, Int): lambda t1, t2: (Int(max(t1.bits, t2.bits)), UInt(1)),
        (Int, UInt): lambda t1, t2: (Int(max(t1.bits, t2.bits + 1)), UInt(1)),
        (Int, Index): lambda t1, t2: (Int(max(t1.bits, t2.bits + 1)), UInt(1)),
        (Int, Fixed): lambda t1, t2: (
            Fixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            UInt(1),
        ),
        (Int, UFixed): lambda t1, t2: (
            Fixed(max(t1.bits, t2.bits - t2.fracs + 1) + t2.fracs, t2.fracs),
            UInt(1),
        ),
        (Int, Float): lambda t1, t2: (t2, UInt(1)),
    }
    uint_rules = {
        (UInt, Int): lambda t1, t2: (Int(max(t1.bits + 1, t2.bits)), UInt(1)),
        (UInt, UInt): lambda t1, t2: (UInt(max(t1.bits, t2.bits)), UInt(1)),
        (UInt, Index): lambda t1, t2: (UInt(max(t1.bits, t2.bits)), UInt(1)),
        (UInt, Fixed): lambda t1, t2: (
            Fixed(max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            UInt(1),
        ),
        (UInt, UFixed): lambda t1, t2: (
            UFixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            UInt(1),
        ),
        (UInt, Float): lambda t1, t2: (t2, UInt(1)),
    }
    index_rules = {
        (Index, Int): lambda t1, t2: (Int(max(t1.bits + 1, t2.bits)), UInt(1)),
        (Index, UInt): lambda t1, t2: (UInt(max(t1.bits, t2.bits)), UInt(1)),
        (Index, Index): lambda t1, t2: (Index(), UInt(1)),
        (Index, Fixed): lambda t1, t2: (
            Fixed(max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            UInt(1),
        ),
        (Index, UFixed): lambda t1, t2: (
            UFixed(max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs),
            UInt(1),
        ),
        (Index, Float): lambda t1, t2: (t2, UInt(1)),
    }
    fixed_rules = {
        (Fixed, Int): lambda t1, t2: (
            Fixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs),
            UInt(1),
        ),
        (Fixed, UInt): lambda t1, t2: (
            Fixed(max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs, t1.fracs),
            UInt(1),
        ),
        (Fixed, Index): lambda t1, t2: (
            Fixed(max(t1.bits - t1.fracs, t2.bits + 1) + t1.fracs, t1.fracs),
            UInt(1),
        ),
        (Fixed, Fixed): lambda t1, t2: (
            Fixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
            UInt(1),
        ),
        (Fixed, UFixed): lambda t1, t2: (
            Fixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs + 1)
                + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
            UInt(1),
        ),
        (Fixed, Float): lambda t1, t2: (t2, UInt(1)),
    }
    ufixed_rules = {
        (UFixed, Int): lambda t1, t2: (
            Fixed(max(t1.bits - t1.fracs + 1, t2.bits) + t1.fracs, t1.fracs),
            UInt(1),
        ),
        (UFixed, UInt): lambda t1, t2: (
            UFixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs),
            UInt(1),
        ),
        (UFixed, Index): lambda t1, t2: (
            UFixed(max(t1.bits - t1.fracs, t2.bits) + t1.fracs, t1.fracs),
            UInt(1),
        ),
        (UFixed, Fixed): lambda t1, t2: (
            Fixed(
                max(t1.bits - t1.fracs + 1, t2.bits - t2.fracs)
                + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
            UInt(1),
        ),
        (UFixed, UFixed): lambda t1, t2: (
            UFixed(
                max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
                max(t1.fracs, t2.fracs),
            ),
            UInt(1),
        ),
        (UFixed, Float): lambda t1, t2: (t2, UInt(1)),
    }
    float_rules = {
        (Float, Int): lambda t1, t2: (t1, UInt(1)),
        (Float, UInt): lambda t1, t2: (t1, UInt(1)),
        (Float, Index): lambda t1, t2: (t1, UInt(1)),
        (Float, Fixed): lambda t1, t2: (t1, UInt(1)),
        (Float, UFixed): lambda t1, t2: (t1, UInt(1)),
        (Float, Float): lambda t1, t2: (Float(max(t1.bits, t2.bits)), UInt(1)),
    }
    return TypeRule(
        ops,
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
    )


def select_rule():
    ops = (ast.SelectOp, ast.Max, ast.Min)
    int_rules = {
        (Int, Int): lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (Int, UInt): lambda t1, t2: Int(max(t1.bits, t2.bits + 1)),
        (Int, Index): lambda t1, t2: Int(max(t1.bits, t2.bits + 1)),
        (Int, Fixed): lambda t1, t2: Fixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (Int, UFixed): lambda t1, t2: Fixed(
            max(t1.bits, t2.bits - t2.fracs + 1) + t2.fracs, t2.fracs
        ),
        (Int, Float): lambda t1, t2: t2,
    }
    uint_rules = {
        (UInt, UInt): lambda t1, t2: UInt(max(t1.bits, t2.bits)),
        (UInt, Index): lambda t1, t2: UInt(max(t1.bits, t2.bits)),
        (UInt, Fixed): lambda t1, t2: Fixed(
            max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (UInt, UFixed): lambda t1, t2: UFixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (UInt, Float): lambda t1, t2: t2,
    }
    index_rules = {
        (Index, Index): lambda t1, t2: Index(),
        (Index, Fixed): lambda t1, t2: Fixed(
            max(t1.bits + 1, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (Index, UFixed): lambda t1, t2: UFixed(
            max(t1.bits, t2.bits - t2.fracs) + t2.fracs, t2.fracs
        ),
        (Index, Float): lambda t1, t2: t2,
    }
    fixed_rules = {
        (Fixed, Fixed): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
            max(t1.fracs, t2.fracs),
        ),
        (Fixed, UFixed): lambda t1, t2: Fixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs + 1) + max(t1.fracs, t2.fracs),
            max(t1.fracs, t2.fracs),
        ),
        (Fixed, Float): lambda t1, t2: t2,
    }
    ufixed_rules = {
        (UFixed, UFixed): lambda t1, t2: UFixed(
            max(t1.bits - t1.fracs, t2.bits - t2.fracs) + max(t1.fracs, t2.fracs),
            max(t1.fracs, t2.fracs),
        ),
        (UFixed, Float): lambda t1, t2: t2,
    }
    float_rules = {
        (Float, Float): lambda t1, t2: Float(max(t1.bits, t2.bits)),
    }
    return TypeRule(
        ops,
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
        commutative=True,
    )


def shift_rule():
    ops = (ast.LeftShiftOp, ast.RightShiftOp)
    int_rules = {
        (Int, Int): lambda t1, t2: t1,
        (Int, UInt): lambda t1, t2: t1,
        (Int, Index): lambda t1, t2: t1,
    }
    uint_rules = {
        (UInt, UInt): lambda t1, t2: t1,
        (UInt, Index): lambda t1, t2: t1,
    }
    index_rules = {
        (Index, Index): lambda t1, t2: Index(),
    }
    return TypeRule(ops, [int_rules, uint_rules, index_rules], commutative=True)


def and_or_rule():
    ops = (ast.And, ast.Or, ast.XOr, ast.Invert)
    int_rules = {
        (Int, Int): lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (Int, UInt): lambda t1, t2: Int(max(t1.bits, t2.bits)),
        (Int, Index): lambda t1, t2: Int(max(t1.bits, t2.bits)),
    }
    uint_rules = {
        (UInt, UInt): lambda t1, t2: UInt(max(t1.bits, t2.bits)),
        (UInt, Index): lambda t1, t2: UInt(max(t1.bits, t2.bits)),
    }
    index_rules = {
        (Index, Index): lambda t1, t2: Index(),
    }
    return TypeRule(ops, [int_rules, uint_rules, index_rules], commutative=True)


def logic_op_rule():
    ops = (ast.LogicalAnd, ast.LogicalOr, ast.LogicalXOr)
    int_rules = {
        (Int, Int): lambda t1, t2: UInt(1),
        (Int, UInt): lambda t1, t2: UInt(1),
        (UInt, UInt): lambda t1, t2: UInt(1),
    }
    return TypeRule(ops, [int_rules])


def pow_rule():
    ops = (ast.MathPowOp,)
    int_rules = {
        (Int, Int): lambda t1, t2: Float(64),
        (Int, UInt): lambda t1, t2: Float(64),
        (Int, Index): lambda t1, t2: Float(64),
        (UInt, UInt): lambda t1, t2: Float(64),
    }
    float_rules = {
        (Float, Float): lambda t1, t2: Float(max(t1.bits, t2.bits)),
        (Float, Int): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
        (Float, UInt): lambda t1, t2: t1 if isinstance(t1, Float) else t2,
    }
    return TypeRule(ops, [int_rules, float_rules])


# TODO: attach typing rules to ast.Operation classes
# Make this a hook? more extensible
def get_type_rules():
    rules = list()
    rules.append(add_sub_rule())
    rules.append(mul_rule())
    rules.append(mod_rule())
    rules.append(and_or_rule())
    rules.append(logic_op_rule())
    rules.append(cmp_rule())
    rules.append(div_rule())
    rules.append(pow_rule())
    rules.append(shift_rule())
    rules.append(select_rule())
    return rules
