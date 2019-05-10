import heterocl as hcl
import numpy as np

def test_schedule_no_return():
    hcl.init()
    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    def algorithm(A, B):
        U = hcl.update(B, lambda x: A[x] + 1)

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    _A = hcl.asarray(np.random.randint(100, size=(10,)), dtype = hcl.Int(32))
    _B = hcl.asarray(np.zeros(10), dtype = hcl.Int(32))

    f(_A, _B)

    _A = _A.asnumpy()
    _B = _B.asnumpy()

    for i in range(10):
        assert(_B[i] == _A[i] + 1)

def test_schedule_return():
    hcl.init()
    A = hcl.placeholder((10,))

    def algorithm(A):
        return hcl.compute(A.shape, lambda x: A[x] + 1)

    s = hcl.create_schedule([A], algorithm)
    f = hcl.build(s)

    _A = hcl.asarray(np.random.randint(100, size=(10,)), dtype = hcl.Int(32))
    _B = hcl.asarray(np.zeros(10), dtype = hcl.Int(32))

    f(_A, _B)

    _A = _A.asnumpy()
    _B = _B.asnumpy()

    for i in range(10):
        assert(_B[i] == _A[i] + 1)

def test_schedule_return_multi():
    hcl.init()
    A = hcl.placeholder((10,))

    def algorithm(A):
        B = hcl.compute(A.shape, lambda x: A[x] + 1)
        C = hcl.compute(A.shape, lambda x: A[x] + 2)
        return B, C

    s = hcl.create_schedule([A], algorithm)
    f = hcl.build(s)

    _A = hcl.asarray(np.random.randint(100, size=(10,)), dtype = hcl.Int(32))
    _B = hcl.asarray(np.zeros(10), dtype = hcl.Int(32))
    _C = hcl.asarray(np.zeros(10), dtype = hcl.Int(32))

    f(_A, _B, _C)

    _A = _A.asnumpy()
    _B = _B.asnumpy()
    _C = _C.asnumpy()

    for i in range(10):
        assert(_B[i] == _A[i] + 1)
        assert(_C[i] == _A[i] + 2)

def test_resize():
    hcl.init()

    def algorithm(A):
        return hcl.compute(A.shape, lambda x: A[x] + 1, "B")

    A = hcl.placeholder((10,), dtype = hcl.UInt(32))

    scheme = hcl.create_scheme([A], algorithm)
    scheme.downsize(algorithm.B, hcl.UInt(2))
    s = hcl.create_schedule_from_scheme(scheme)
    f = hcl.build(s)

    a = np.random.randint(100, size=(10,))
    _A = hcl.asarray(a, dtype = hcl.UInt(32))
    _B = hcl.asarray(np.zeros(10), dtype = hcl.UInt(2))

    f(_A, _B)

    _A = _A.asnumpy()
    _B = _B.asnumpy()

    for i in range(10):
        assert(_B[i] == (a[i] + 1)%4)

def test_select():
    hcl.init(hcl.Float())
    A = hcl.placeholder((10,))
    B = hcl.compute(A.shape, lambda x: hcl.select(A[x] > 0.5, A[x], 0))
    s = hcl.create_schedule([A, B])
    f = hcl.build(s)

    np_A = np.random.rand(10)
    np_B = np.zeros(10)
    np_C = np.zeros(10)

    for i in range(0, 10):
        np_C[i] = np_A[i] if np_A[i] > 0.5 else 0

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.allclose(np_B, np_C)
