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

def test_simplifier():
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
        upper_idx = a.v >> 1
        B = hcl.compute(A.shape, lambda x: A[x][lower_idx:upper_idx])
        return B
    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b101101, 0b101110])
    np_B = hcl.asarray([0, 0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b101101, 0b101110]

# pytest does not execute this part
# you can run this file directly to test
# e.g. python test_simplify.py
if __name__ == "__main__":
    test_simplifier()