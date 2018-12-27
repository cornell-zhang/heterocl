import heterocl as hcl
import numpy as np

def test_reduce_basic():

    def kernel(A):
        my_sum = hcl.reducer(0, lambda x, y: x+y)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute((1,), lambda x: my_sum(A[r], axis=r))

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(1)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A)
    assert ret_B[0] == golden_B

def test_reduce_cond():

    def kernel(A):
        my_sum = hcl.reducer(0, lambda x, y: x+y)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute((1,), lambda x: my_sum(A[r], axis=r, where=A[r]>5))

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(1)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A[np.where(np_A > 5)])
    assert ret_B[0] == golden_B

def test_reduce_dtype():

    def kernel(A):
        # my_sum will perform integer reduction
        my_sum = hcl.reducer(0, lambda x, y: x+y)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute((1,), lambda x: my_sum(A[r], axis=r, dtype=hcl.Float()), dtype=hcl.Float())

    A = hcl.placeholder((10,), dtype=hcl.Float())
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.rand(10)
    np_B = np.zeros(1)

    hcl_A = hcl.asarray(np_A, dtype=hcl.Float())
    hcl_B = hcl.asarray(np_B, dtype=hcl.Float())

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A)
    assert np.isclose(ret_B[0], golden_B)

def test_reduce_dtype_2():

    def kernel(A):
        my_sum = hcl.reducer(0, lambda x, y: x+y)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute((1,), lambda x: my_sum(A[r], axis=r, dtype=hcl.UInt(2)))

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(1)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A)
    assert ret_B[0] == golden_B%4

def test_reduce_different_init():

    def kernel(a, A):
        my_sum = hcl.reducer(a, lambda x, y: x+y)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute((1,), lambda x: my_sum(A[r], axis=r))

    a = hcl.placeholder(())
    A = hcl.placeholder((10,))
    s = hcl.create_schedule([a, A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(1)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(10, hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A)
    assert ret_B[0] == golden_B+10

def test_reduce_2D():

    def kernel(A):
        my_sum = hcl.reducer(0, lambda x, y: x+y)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute((10,), lambda x: my_sum(A[x, r], axis=r))

    A = hcl.placeholder((10, 10))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 10))
    np_B = np.zeros(10)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A, axis=1)
    assert np.array_equal(ret_B, golden_B)

def test_reduce_multi_axes():

    def kernel(A):
        my_sum = hcl.reducer(0, lambda x, y: x+y)
        r1 = hcl.reduce_axis(0, 10)
        r2 = hcl.reduce_axis(0, 10)
        return hcl.compute((1,), lambda x: my_sum(A[r1, r2], axis=[r1, r2]))

    A = hcl.placeholder((10, 10))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 10))
    np_B = np.zeros(1)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A)
    assert ret_B[0] == golden_B

def test_reduce_complex_reducer():

    def kernel(A):
        def reducer_body(x, y):
            with hcl.if_(x > 5):
                hcl.return_(y + 1)
            with hcl.else_():
                hcl.return_(y + 2)
        my_sum = hcl.reducer(0, reducer_body)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute((1,), lambda x: my_sum(A[r], axis=r))

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(1)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = 10 + np.sum(len(np_A[np.where(np_A <= 5)]))
    assert ret_B[0] == golden_B

