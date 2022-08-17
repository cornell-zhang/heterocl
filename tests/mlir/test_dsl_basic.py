import heterocl as hcl
import numpy as np
import pytest
from hcl_mlir.exceptions import MLIRLimitationError


def _test_logic_op(op):
    def kernel(A, B):
        return hcl.compute(A.shape, lambda x: hcl.select(op(A[x] > 5, B[x] > 5), 0, 1))

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    return f


def test_and():

    f = _test_logic_op(hcl.and_)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.random.randint(10, size=(10,))
    np_C = np.zeros(10)

    golden_C = [0 if np_A[i] > 5 and np_B[i] > 5 else 1 for i in range(0, 10)]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_C = hcl.asarray(np_C)

    f(hcl_A, hcl_B, hcl_C)

    ret_C = hcl_C.asnumpy()
    assert np.array_equal(ret_C, golden_C)

def test_or():

    with pytest.raises(MLIRLimitationError):
        f = _test_logic_op(hcl.or_)

        np_A = np.random.randint(10, size=(10,))
        np_B = np.random.randint(10, size=(10,))
        np_C = np.zeros(10)

        golden_C = [0 if np_A[i] > 5 or np_B[i] > 5 else 1 for i in range(0, 10)]

        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B)
        hcl_C = hcl.asarray(np_C)

        f(hcl_A, hcl_B, hcl_C)

        ret_C = hcl_C.asnumpy()
        assert np.array_equal(ret_C, golden_C)

def test_if():
    def kernel(A):
        with hcl.if_(A[0] > 5):
            A[0] = 5

    A = hcl.placeholder((1,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(1,))
    golden_A = [5 if np_A[0] > 5 else np_A[0]]

    hcl_A = hcl.asarray(np_A)

    f(hcl_A)

    ret_A = hcl_A.asnumpy()
    assert np.array_equal(golden_A, ret_A)

def test_else():
    hcl.init(hcl.Int(32))
    def kernel(A):
        with hcl.if_(A[0] > 5):
            A[0] = 5
        with hcl.else_():
            A[0] = -1

    A = hcl.placeholder((1,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)
    np_A = np.random.randint(10, size=(1,))
    golden_A = [5 if np_A[0] > 5 else -1]
    hcl_A = hcl.asarray(np_A)
    f(hcl_A)
    ret_A = hcl_A.asnumpy()
    assert np.array_equal(golden_A, ret_A)

def test_cond_all():
    def kernel(A):
        with hcl.if_(A[0] > 5):
            A[0] = 5
        with hcl.elif_(A[0] > 3):
            A[0] = 3
        with hcl.else_():
            A[0] = 0

    A = hcl.placeholder((1,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(1,))
    golden_A = [5 if np_A[0] > 5 else (3 if np_A[0] > 3 else 0)]

    hcl_A = hcl.asarray(np_A)

    f(hcl_A)

    ret_A = hcl_A.asnumpy()

def test_elif():
    def kernel(A):
        with hcl.if_(A[0] > 5):
            A[0] = 5
        with hcl.elif_(A[0] > 3):
            A[0] = 3

    A = hcl.placeholder((1,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(1,))
    golden_A = [5 if np_A[0] > 5 else (3 if np_A[0] > 3 else np_A[0])]

    hcl_A = hcl.asarray(np_A)

    f(hcl_A)

    ret_A = hcl_A.asnumpy()
    assert np.array_equal(golden_A, ret_A)


def test_for_basic():
    def kernel(A):
        with hcl.for_(0, 10) as i:
            A[i] = i

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    golden_A = [i for i in range(0, 10)]

    hcl_A = hcl.asarray(np_A)

    f(hcl_A)

    ret_A = hcl_A.asnumpy()
    assert np.array_equal(golden_A, ret_A)


def test_for_irregular_bound():
    def kernel(A):
        with hcl.for_(4, 8) as i:
            A[i] = i

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    golden_A = np.copy(np_A)
    for i in range(4, 8):
        golden_A[i] = i

    hcl_A = hcl.asarray(np_A)

    f(hcl_A)

    ret_A = hcl_A.asnumpy()
    assert np.array_equal(golden_A, ret_A)


def test_for_step_non_one():
    def kernel(A):
        with hcl.for_(0, 10, 2) as i:
            A[i] = i

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    golden_A = np.copy(np_A)
    for i in range(0, 10, 2):
        golden_A[i] = i

    hcl_A = hcl.asarray(np_A)

    f(hcl_A)

    ret_A = hcl_A.asnumpy()
    assert np.array_equal(golden_A, ret_A)


def test_for_step_negative():
    def kernel(A):
        with hcl.for_(9, -1, -1) as i:
            A[i] = i

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    golden_A = [i for i in range(0, 10)]

    hcl_A = hcl.asarray(np_A)

    f(hcl_A)

    ret_A = hcl_A.asnumpy()
    assert np.array_equal(golden_A, ret_A)


def test_for_index_casting():
    def kernel(A):
        with hcl.for_(0, 10) as i:
            with hcl.for_(i, 10) as j:
                A[j] += i

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.zeros(10)
    golden_A = np.zeros(10)

    for i in range(0, 10):
        for j in range(i, 10):
            golden_A[j] += i

    hcl_A = hcl.asarray(np_A)

    f(hcl_A)

    ret_A = hcl_A.asnumpy()
    assert np.array_equal(golden_A, ret_A)


def test_while_basic():
    def kernel(A):
        a = hcl.scalar(0)
        with hcl.while_(a[0] < 10):
            A[a[0]] = a[0]
            a[0] += 1

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    golden_A = [i for i in range(0, 10)]

    hcl_A = hcl.asarray(np_A)

    f(hcl_A)

    ret_A = hcl_A.asnumpy()
    assert np.array_equal(golden_A, ret_A)

def test_invalidate():
    def kernel():
        i = hcl.scalar(0,"i",dtype=hcl.UInt(32))
        with hcl.while_(i.v < 10):
            i.v += 1
    s = hcl.create_schedule([], kernel)

def test_break_in_for():
    with pytest.raises(Exception):
        def kernel(A):
            with hcl.for_(0, 10) as i:
                with hcl.if_(i > 5):
                    hcl.break_()
                A[i] = i

        A = hcl.placeholder((10,))
        s = hcl.create_schedule(A, kernel)
        f = hcl.build(s)

        np_A = np.random.randint(10, size=(10,))
        golden_A = np.copy(np_A)
        for i in range(0, 6):
            golden_A[i] = i

        hcl_A = hcl.asarray(np_A)

        f(hcl_A)

        ret_A = hcl_A.asnumpy()
        assert np.array_equal(golden_A, ret_A)


def test_break_in_while():
    with pytest.raises(Exception):
        def kernel(A):
            i = hcl.scalar(0)
            with hcl.while_(True):
                with hcl.if_(i[0] > 5):
                    hcl.break_()
                A[i[0]] = i[0]
                i[0] += 1

        A = hcl.placeholder((10,))
        s = hcl.create_schedule(A, kernel)
        f = hcl.build(s)

        np_A = np.random.randint(10, size=(10,))
        golden_A = np.copy(np_A)
        for i in range(0, 6):
            golden_A[i] = i

        hcl_A = hcl.asarray(np_A)

        f(hcl_A)

        ret_A = hcl_A.asnumpy()
        assert np.array_equal(golden_A, ret_A)


def test_break_multi_level():
    with pytest.raises(Exception):
        def kernel(A):
            with hcl.for_(0, 10) as i:
                with hcl.for_(0, 10) as j:
                    with hcl.if_(j >= i):
                        hcl.break_()
                    A[i] += j

        A = hcl.placeholder((10,))
        s = hcl.create_schedule(A, kernel)
        f = hcl.build(s)

        np_A = np.random.randint(10, size=(10,))
        golden_A = np.copy(np_A)
        for i in range(0, 10):
            for j in range(0, i):
                golden_A[i] += j

        hcl_A = hcl.asarray(np_A)

        f(hcl_A)

        ret_A = hcl_A.asnumpy()
        assert np.array_equal(golden_A, ret_A)


def test_get_bit_expr():

    hcl.init()

    def kernel(A):
        return hcl.compute(A.shape, lambda x: (A[x] + 1)[0])

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(10)
    golden = (np_A + 1) & 1
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)


def test_get_bit_tensor():

    hcl.init()

    def kernel(A):
        return hcl.compute(A.shape, lambda x: A[x][0])

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(10)
    golden = np_A & 1
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)


def test_set_bit_expr():

    hcl.init()

    def kernel(A, B):
        with hcl.for_(0, 10) as i:
            (B[i] + 1)[0] = A[i]

    A = hcl.placeholder((10,), dtype=hcl.Int(1))
    B = hcl.placeholder((10,))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.lower(s)
    assert "set_bit" in str(f)


def test_set_bit_tensor():

    hcl.init()

    def kernel(A, B):
        with hcl.for_(0, 10) as i:
            B[i][0] = A[i]

    A = hcl.placeholder((10,), dtype=hcl.Int(1))
    B = hcl.placeholder((10,))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(1, size=(10,))
    np_B = np.random.randint(10, size=(10,))
    golden = (np_B & 0b1110) | np_A
    hcl_A = hcl.asarray(np_A, dtype=hcl.Int(1))
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)


def test_get_slice_expr():

    hcl.init()

    def kernel(A):
        return hcl.compute(A.shape, lambda x: (A[x] + 1)[0:2])

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(10)
    golden = (np_A + 1) & 0b11
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    print(ret)
    print(golden)
    assert np.array_equal(golden, ret)


def test_get_slice_tensor():

    hcl.init()

    def kernel(A):
        return hcl.compute(A.shape, lambda x: A[x][0:2])

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(10)
    golden = np_A & 0b11
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)


def test_get_slice_tensor_reverse():

    hcl.init(hcl.UInt(8))

    def kernel(A):
        return hcl.compute(A.shape, lambda x: (A[x][0:8]).reverse())

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(10)
    golden = np_A & 0xFF
    golden = golden.astype("uint8")
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    ret = ret.astype("uint8")

    for i in range(0, 10):
        x = np.unpackbits(golden[i])
        x = np.flip(x)
        y = np.unpackbits(ret[i])
        assert np.array_equal(x, y)


def test_set_slice_expr():

    hcl.init()

    def kernel(A, B):
        with hcl.for_(0, 10) as i:
            (B[i] + 1)[0:2] = A[i]

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.lower(s)
    assert "set_slice" in str(f)


def test_set_slice_tensor():

    hcl.init()

    def kernel(A, B):
        with hcl.for_(0, 10) as i:
            B[i][0:2] = A[i]

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(1, size=(10,))
    print("np_A: ")
    print(np_A, end="\n\n")
    np_B = np.random.randint(10, size=(10,))
    print("np_B: ")
    print(np_B, end="\n\n")
    golden = (np_B & 0b1100) | np_A
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    print("golden: ")
    print(golden)
    print("result: ")
    print(ret)
    assert np.array_equal(golden, ret)


def test_set_slice_tensor_reverse():

    hcl.init(hcl.UInt(8))

    def kernel(A, B):
        with hcl.for_(0, 10) as i:
            B[i][0:8] = A[i].reverse()

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(1, size=(10,))
    np_B = np.random.randint(10, size=(10,))
    np_A = np_A.astype("uint8")
    np_B = np_B.astype("uint8")
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    ret = ret.astype("uint8")

    for i in range(0, 10):
        a = np.flip(np.unpackbits(np_A[i]))
        b = np.unpackbits(ret[i])
        assert np.array_equal(a, b)


def test_slice_op():

    hcl.init()

    def kernel(A):
        return hcl.compute(A.shape, lambda x: A[x][0:8] + A[x][8:16])

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(10)
    golden = (np_A & 0xFF) + ((np_A >> 8) & 0xFF)
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)


def test_tensor_slice():
    hcl.init()
    def kernel():
        z1 = hcl.compute((2,3,4), lambda x,y,z: 0, dtype=hcl.Int(32))
        def do(i,j,k):
            z1[i][j][k] = i + j + k
        hcl.mutate(z1.shape, do)
        return z1
    
    s = hcl.create_schedule([], kernel)
    hcl_res = hcl.asarray(np.zeros((2,3,4), dtype=np.int32))
    f = hcl.build(s)
    f(hcl_res)
    np_res = hcl_res.asnumpy()
    golden = np.zeros((2,3,4), dtype=np.int32)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                golden[i][j][k] = i + j + k
    assert np.array_equal(golden, np_res)
