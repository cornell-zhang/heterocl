import heterocl as hcl
import numpy as np

def test_reduce_basic():
    hcl.init()

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
        return hcl.compute((10,), lambda x: my_sum(A[r, x], axis=r))

    A = hcl.placeholder((10, 10))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 10))
    np_B = np.zeros(10)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A, axis=0)
    assert np.array_equal(ret_B, golden_B)

def test_reduce_2D_2():

    def kernel(A):
        my_sum = hcl.reducer(0, lambda x, y: x+y)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute((1, 10), lambda x, y: my_sum(A[r, y], axis=r))

    A = hcl.placeholder((10, 10))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 10))
    np_B = np.zeros((1, 10))

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A, axis=0)
    assert np.array_equal(ret_B[0], golden_B)

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
    golden_B = 10 + len(np_A[np.where(np_A <= 5)])
    assert ret_B[0] == golden_B

def test_reduce_tensor():

    def kernel(A):
        init = hcl.compute((2,), lambda x: 10)
        def freduce(x, Y):
            with hcl.if_(x < Y[0]):
                Y[1] = Y[0]
                Y[0] = x
            with hcl.else_():
                with hcl.if_(x < Y[1]):
                    Y[1] = x
        my_min = hcl.reducer(init, freduce)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute((2,), lambda _x: my_min(A[r], axis=r))

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(2)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sort(np_A)[:2]
    assert np.array_equal(ret_B, golden_B)

def test_reduce_sort():

    def kernel(A):
        init = hcl.compute(A.shape, lambda x: 11)
        def freduce(x, Y):
            with hcl.for_(0, 10) as i:
                with hcl.if_(x < Y[i]):
                    with hcl.for_(9, i, -1) as j:
                        Y[j] = Y[j-1]
                    Y[i] = x
                    hcl.break_()
        my_sort = hcl.reducer(init, freduce)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute(A.shape, lambda _x: my_sort(A[r], axis=r))

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(10)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sort(np_A)
    assert np.array_equal(ret_B, golden_B)

def test_reduce_sort_2D():

    def kernel(A):
        init = hcl.compute((10,), lambda x: 11)
        def freduce(x, Y):
            with hcl.for_(0, 10) as i:
                with hcl.if_(x < Y[i]):
                    with hcl.for_(9, i, -1) as j:
                        Y[j] = Y[j-1]
                    Y[i] = x
                    hcl.break_()
        my_sort = hcl.reducer(init, freduce)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute(A.shape, lambda _x, y: my_sort(A[r, y], axis=r))

    A = hcl.placeholder((10, 10))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 10))
    np_B = np.zeros((10, 10))

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sort(np_A, axis=0)
    assert np.array_equal(ret_B, golden_B)

def test_reduce_sort_2D_flatten():

    def kernel(A):
        init = hcl.compute((A.shape[0]*A.shape[1],), lambda x: 11)
        def freduce(x, Y):
            with hcl.for_(0, Y.shape[0]) as i:
                with hcl.if_(x < Y[i]):
                    with hcl.for_(Y.shape[0]-1, i, -1) as j:
                        Y[j] = Y[j-1]
                    Y[i] = x
                    hcl.break_()
        my_sort = hcl.reducer(init, freduce)
        rx = hcl.reduce_axis(0, 10)
        ry = hcl.reduce_axis(0, 10)
        return hcl.compute(init.shape, lambda _x: my_sort(A[rx, ry], axis=[rx, ry]))

    A = hcl.placeholder((10, 10))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 10))
    np_B = np.zeros(100)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sort(np_A, axis=None)
    assert np.array_equal(ret_B, golden_B)
