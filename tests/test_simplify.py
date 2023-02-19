import heterocl as hcl
import pytest
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
        upper_idx = 1 << a.v # this doesn't work now because simplifier doesn't support shift
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
        upper_idx = a.v >> 1 # 12 >> 1 = 5

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

        lower_idx = 1 & a.v # 1 & 21 = 1
        upper_idx = a.v & b.v # 21 & 13 = 5

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

        lower_idx = a.v | 1 # 2 | 1 = 3
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
        upper_idx = a.v ^ b.v # 5 ^ 2 = 7
        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B
    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b01100101, 0b10010011])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b11001, 0b00100]

def test_cmp_lt():
    def kernel(A):
        a = hcl.scalar(5)

        lower_idx = 0
        with hcl.if_(a.v < 8): # TODO: It goes through both the if and else branches
            upper_idx = 5
        with hcl.else_():
            upper_idx = 3
        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B
    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b1101001010, 0b0111001110])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b01010, 0b01110]


# pytest does not execute this part
# you can run this file directly to test
# e.g. python test_simplify.py
if __name__ == "__main__":
    def kernel(A):
        a = hcl.scalar(5)

        lower_idx = 0
        upper_idx = 2 < a.v
        print(upper_idx)
        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B
    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b1101001010, 0b0111001110])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b01010, 0b01110]