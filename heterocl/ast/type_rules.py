# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
""" Type inference rules """

from ..types import Index, Float, Int, UInt, Fixed, UFixed, TypeRule

# TODO: Reduction op rules


def add_sub_rule():
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
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
    )


def mul_rule():
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
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
        commutative=True,
    )


def div_rule():
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
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
    )


def mod_rule():
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
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
    )


def cmp_rule():
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
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
    )


def select_rule():
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
        [int_rules, uint_rules, index_rules, fixed_rules, ufixed_rules, float_rules],
        commutative=True,
    )


def shift_rule():
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
    return TypeRule([int_rules, uint_rules, index_rules], commutative=True)


def and_or_rule():
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
    return TypeRule([int_rules, uint_rules, index_rules], commutative=True)


def logic_op_rule():
    int_rules = {
        (Int, Int): lambda t1, t2: UInt(1),
        (Int, UInt): lambda t1, t2: UInt(1),
        (UInt, UInt): lambda t1, t2: UInt(1),
    }
    return TypeRule([int_rules])


def pow_rule():
    def select_float(t1, _):
        if t1.bits <= 32:
            return Float(32)
        return Float(64)

    int_rule = {
        (Int, Int): select_float,
        (Int, UInt): select_float,
        (Int, Index): select_float,
        (Int, Fixed): select_float,
        (Int, UFixed): select_float,
        (Int, Float): select_float,
    }
    uint_rule = {
        (UInt, UInt): select_float,
        (UInt, Index): select_float,
        (UInt, Fixed): select_float,
        (UInt, UFixed): select_float,
        (UInt, Float): select_float,
    }
    index_rule = {
        (Index, Index): select_float,
        (Index, Fixed): select_float,
        (Index, UFixed): select_float,
        (Index, Float): select_float,
    }
    fixed_rule = {
        (Fixed, Fixed): select_float,
        (Fixed, UFixed): select_float,
        (Fixed, Float): select_float,
    }
    ufixed_rule = {(UFixed, UFixed): select_float, (UFixed, Float): select_float}
    float_rule = {
        (Float, Float): lambda t1, t2: Float(max(t1.bits, t2.bits)),
    }
    # Commutative=True here doesn't mean that power operation is commutative.
    # It means that the type rule is commutative, to reduce the number of rules.
    # e.g. hcl.power(a, b) and hcl.power(b, a) will have the same type rule.
    # because MLIR math op in LLVM 15 only has float pow op.
    return TypeRule(
        [int_rule, uint_rule, index_rule, fixed_rule, ufixed_rule, float_rule],
        commutative=True,
    )


def intrin_rule():
    # covers:
    # expr, log, log2, log10, sqrt,
    # sin, cos, tanh
    unaryrules = {
        (Float,): lambda t: t,
        (Int,): lambda t: Float(32) if t.bits <= 32 else Float(64),
        (UInt,): lambda t: Float(32) if t.bits <= 32 else Float(64),
        (Index,): lambda t: Float(32) if t.bits <= 32 else Float(64),
        (Fixed,): lambda t: Float(32) if t.bits <= 32 else Float(64),
        (UFixed,): lambda t: Float(32) if t.bits <= 32 else Float(64),
    }
    return TypeRule([unaryrules])
