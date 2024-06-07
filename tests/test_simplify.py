# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import pytest
import numpy as np
from hcl_mlir.exceptions import APIError


def test_remove_single_loop():
    hcl.init()
    a = hcl.placeholder((1,))
    b = hcl.compute(a.shape, lambda x: a[x] + 1)
    s = hcl.create_schedule([a])
    ir = hcl.lower(s)
    assert "0 to 1" not in str(ir)


def test_simplify_slice():
    with pytest.raises(APIError):
        hcl.init()
        A = hcl.placeholder((10,), "A")

        def kernel(A):
            A[5][2:2] = 4

        s = hcl.create_schedule(A, kernel)
        ir = hcl.lower(s)


def test_left_shift_op():
    # in this example, we use a scalar's value
    # to calculate the lower and upper index
    # for integer slicing.
    # for example, if the input is 0b101101,
    # when we slice it with [0:4], we will get
    # 0b1101, the first four bits, which is 11 in decimal.
    def kernel(A):
        a = hcl.scalar(2)
        lower_idx = 0

        # upper index is 1 << 2, which is 4
        upper_idx = (
            1 << a.v
        )  # this doesn't work now because simplifier doesn't support shift
        # if we change upper_idx to 4, this test will pass, but it's not what we want

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b101101, 0b101110])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b1101, 0b1110]


def test_right_shift_op():
    def kernel(A):
        a = hcl.scalar(12)

        lower_idx = 0
        upper_idx = a.v >> 1  # 12 >> 1 = 5

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b101101, 0b101110])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b101101, 0b101110]


def test_bitwise_and():
    def kernel(A):
        a = hcl.scalar(21)
        b = hcl.scalar(13)

        lower_idx = 1 & a.v  # 1 & 21 = 1
        upper_idx = a.v & b.v  # 21 & 13 = 5

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b01100101, 0b10010011])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b0010, 0b1001]


def test_bitwise_or():
    def kernel(A):
        a = hcl.scalar(2)

        lower_idx = a.v | 1  # 2 | 1 = 3
        upper_idx = 8

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b01100101, 0b10010011])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b01100, 0b10010]


def test_bitwise_xor():
    def kernel(A):
        a = hcl.scalar(5)
        b = hcl.scalar(2)

        lower_idx = b.v
        upper_idx = a.v ^ b.v  # 5 ^ 2 = 7

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b01100101, 0b10010011])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b11001, 0b00100]


def test_cmp_lt1():
    def kernel(A):
        a = hcl.scalar(5)

        lower_idx = 0
        upper_idx = 8 > a.v  # 8 > 5 -> 5 < 8 = True -> 1

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b10, 0b01])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b0, 0b1]


def test_cmp_le1():
    def kernel(A):
        a = hcl.scalar(5)

        lower_idx = 0
        upper_idx = 5 >= a.v  # 5 >= 5 -> 5 <= 5 = True -> 1

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b10, 0b01])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b0, 0b1]


def test_cmp_eq1():
    def kernel(A):
        a = hcl.scalar(5)

        lower_idx = a.v == 6  # 5 == 6 -> 6 == 5 = False -> 0
        upper_idx = 5 == a.v  # 5 == 5 = True -> 1

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b11, 0b01])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b1, 0b1]


def test_cmp_ne1():
    def kernel(A):
        a = hcl.scalar(12)

        lower_idx = 12 != a.v  # 12 != 12 = False -> 0
        upper_idx = 5 != a.v  # 5 != 12 -> 12 != 5 = True -> 1

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b00, 0b10])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b0, 0b0]


def test_cmp_gt1():
    def kernel(A):
        a = hcl.scalar(5)

        lower_idx = 0
        upper_idx = 2 < a.v  # 2 < 5 -> 5 > 2 = True -> 1

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b1101001011, 0b0111001110])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b1, 0b0]


def test_cmp_ge1():
    def kernel(A):
        a = hcl.scalar(5)

        lower_idx = 0
        upper_idx = 2 <= a.v  # 2 <= 5 -> 5 => 2 = True -> 1

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b1101001011, 0b0111001110])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b1, 0b0]


def test_struct_get_op():
    def kernel(A):
        stype = hcl.Struct({"binary": hcl.Int(8), "upper_idx": hcl.Int(8)})

        lower_idx = 0

        B = hcl.compute(A.shape, lambda x: (A[x], 5), dtype=stype)
        C = hcl.compute(A.shape, lambda x: (B[x].binary)[lower_idx : (B[x].upper_idx)])
        return C

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b11010110, 0b10011100])
    np_C = hcl.asarray([0, 0])
    f(np_A, np_C)
    assert np_C.asnumpy().tolist() == [0b10110, 0b11100]


def test_select_op1():
    def kernel(A):
        c1 = hcl.scalar(10)
        c2 = hcl.scalar(4)

        lower_idx = 0
        upper_idx = hcl.select(5 < c1.v, c2.v, 6)

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b01101010, 0b11101011])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b1010, 0b1011]


def test_select_op2():
    def kernel(A):
        c1 = hcl.scalar(10)
        c2 = hcl.scalar(4)

        lower_idx = 0
        upper_idx = hcl.select(5 < c2.v, c1.v, 6)

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b01101010, 0b11101011])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b101010, 0b101011]


def test_logical_and():
    def kernel(A):
        a = hcl.scalar(1)
        b = hcl.scalar(0)

        lower_idx = hcl.and_(a.v, b.v)
        upper_idx = hcl.and_(a.v, a.v)

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b11, 0b10])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b1, 0b0]


def test_logical_or():
    def kernel(A):
        a = hcl.scalar(1)
        b = hcl.scalar(0)

        lower_idx = hcl.or_(b.v, b.v)
        upper_idx = hcl.or_(a.v, b.v)

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b11, 0b10])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b1, 0b0]


def test_neg():
    def kernel(A):
        a = hcl.scalar(-5)

        lower_idx = 0

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx : (-a.v)])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b10110101, 0b10001110])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b10101, 0b01110]


def test_math_exp():
    def kernel(A):
        lower_idx = hcl.exp(0)
        lower_idx = hcl.cast(hcl.Index(), lower_idx)
        upper_idx = hcl.exp(1.5)
        upper_idx = hcl.cast(hcl.Index(), upper_idx)  # e^1.5 ~4.482 -> 4

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b10101100, 0b01100101])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b110, 0b010]


def test_math_pow():
    def kernel(A):
        a = hcl.scalar(0)
        b = hcl.scalar(2)

        lower_idx = hcl.power(b.v, a.v)
        lower_idx = hcl.cast(hcl.Index(), lower_idx)
        upper_idx = hcl.power(b.v, b.v)
        upper_idx = hcl.cast(hcl.Index(), upper_idx)

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b10101100, 0b01100101])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b110, 0b010]


def test_math_log_op():
    def kernel(A):
        a = hcl.scalar(1)

        lower_idx = hcl.log(a.v)
        lower_idx = hcl.cast(hcl.Index(), lower_idx)
        upper_idx = hcl.log(12)
        upper_idx = hcl.cast(hcl.Index(), upper_idx) + 3

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b01001111, 0b11101110])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b01111, 0b01110]


def test_math_log2_op():
    def kernel(A):
        a = hcl.scalar(16)

        lower_idx = hcl.log2(2)
        lower_idx = hcl.cast(hcl.Index(), lower_idx)
        upper_idx = hcl.log2(a.v)
        upper_idx = hcl.cast(hcl.Index(), upper_idx)

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b00001010, 0b11101110])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b101, 0b111]


def test_math_log10_op():
    def kernel(A):
        a = hcl.scalar(10)
        b = hcl.scalar(10000)

        lower_idx = hcl.log10(a.v)
        lower_idx = hcl.cast(hcl.Index(), lower_idx)
        upper_idx = hcl.log10(b.v)
        upper_idx = hcl.cast(hcl.Index(), upper_idx)

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b01111100, 0b01001010])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b110, 0b101]


def test_math_sqrt1():
    def kernel(A):
        a = hcl.scalar(25)

        upper_idx = hcl.sqrt(a.v)
        upper_idx = hcl.cast(hcl.Index(), upper_idx)

        B = hcl.compute(A.shape, lambda x: A[x][0:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b10011101, 0b01001010])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b11101, 0b01010]


def test_math_sqrt2():
    def kernel(A):
        upper_idx = hcl.sqrt(30)
        upper_idx = hcl.cast(hcl.Index(), upper_idx)  # sqrt(30) ~ 5.477 -> 5

        B = hcl.compute(A.shape, lambda x: A[x][0:upper_idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b10011101, 0b01001010])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b11101, 0b01010]


def test_math_sin1():
    def kernel(A):
        lower_idx = hcl.sin(0)
        lower_idx = hcl.cast(hcl.Index(), lower_idx)

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:3])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b10011101, 0b10110001])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b101, 0b001]


def test_math_sin2():
    def kernel(A):
        lower_idx = hcl.sin(np.pi / 2)
        lower_idx = hcl.cast(hcl.Index(), lower_idx)

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:5])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b10011101, 0b10110001])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b1110, 0b1000]


def test_math_cos1():
    def kernel(A):
        lower_idx = hcl.cos(0)
        lower_idx = hcl.cast(hcl.Index(), lower_idx)

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:4])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b10011101, 0b10110001])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b110, 0b000]


def test_math_cos2():
    def kernel(A):
        lower_idx = hcl.cos(np.pi / 2)
        lower_idx = hcl.cast(hcl.Index(), lower_idx)

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:4])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b10011101, 0b10110001])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b1101, 0b0001]


# hcl.tan() not supported yet

# def test_math_tan():
#     def kernel(A):
#         lower_idx = hcl.tan(0)
#         lower_idx = hcl.cast(hcl.Index(), lower_idx)

#         B = hcl.compute(A.shape, lambda x: A[x][lower_idx:4])
#         return B

#     A = hcl.placeholder((2,), "A")
#     s = hcl.create_schedule([A], kernel)
#     f = hcl.build(s)
#     np_A = hcl.asarray([0b10011101, 0b00101100])
#     np_B = hcl.asarray([0, 0])
#     f(np_A, np_B)
#     assert np_B.asnumpy().tolist() == [0b1101, 0b1100]


def test_math_tanh1():
    def kernel(A):
        lower_idx = hcl.tanh(0)
        lower_idx = hcl.cast(hcl.Index(), lower_idx)

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:4])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b10101010, 0b11001011])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b1010, 0b1011]


def test_math_tanh2():
    def kernel(A):
        a = hcl.scalar(8)

        lower_idx = hcl.tanh(a.v)
        lower_idx = hcl.cast(hcl.Index(), lower_idx)

        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:4])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b10101011, 0b11001011])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b101, 0b101]
