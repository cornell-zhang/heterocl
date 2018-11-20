import heterocl as hcl
import numpy as np

"""
Testing API: module
"""
def test_module_no_return():

    def algorithm(A, B):

        @hcl.module([A.shape, B.shape, ()])
        def update_B(A, B, x):
            B[x] = A[x] + 1

        with hcl.Stage():
            with hcl.for_(0, 10) as i:
                update_B(A, B, i)

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    a = np.random.randint(100, size=(10,))
    b = np.zeros(10)
    _A = hcl.asarray(a)
    _B = hcl.asarray(b, hcl.Int())

    f(_A, _B)

    _A = _A.asnumpy()
    _B = _B.asnumpy()

    for i in range(0, 10):
        assert(_B[i] == a[i]+1)


def test_module_with_return():

    def algorithm(A, B):

        @hcl.module([A.shape, ()])
        def update_B(A, x):
            hcl.return_(A[x] + 1)

        hcl.update(B, lambda x: update_B(A, x))

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    a = np.random.randint(100, size=(10,))
    b = np.zeros(10)
    _A = hcl.asarray(a)
    _B = hcl.asarray(b, hcl.Int())

    f(_A, _B)

    _A = _A.asnumpy()
    _B = _B.asnumpy()

    for i in range(0, 10):
        assert(_B[i] == a[i]+1)

def test_module_multi_calls():

    def algorithm(A, B):

        @hcl.module([A.shape, B.shape, ()])
        def add(A, B, x):
            hcl.return_(A[x] + B[x])

        @hcl.module([A.shape, B.shape, ()])
        def mul(A, B, x):
            temp = hcl.local(0)
            with hcl.for_(0, x) as i:
                temp[0] += add(A, B, x)
            hcl.return_(temp[0])

        return hcl.compute(A.shape, lambda x: mul(A, B, x))

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    a = np.random.randint(100, size=(10,))
    b = np.random.randint(100, size=(10,))
    c = np.zeros(10)
    _A = hcl.asarray(a)
    _B = hcl.asarray(b)
    _C = hcl.asarray(c, hcl.Int())

    f(_A, _B, _C)

    _A = _A.asnumpy()
    _B = _B.asnumpy()
    _C = _C.asnumpy()

    for i in range(0, 10):
        assert(_C[i] == (a[i]+b[i])*i)

def test_module_ret_dtype():

    def algorithm(A, B):

        @hcl.module([A.shape, B.shape, ()], ret_dtype=hcl.UInt(2))
        def add(A, B, x):
            hcl.return_(A[x] + B[x])

        return hcl.compute(A.shape, lambda x: add(A, B, x), dtype=hcl.UInt(2))

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    a = np.random.randint(100, size=(10,))
    b = np.random.randint(100, size=(10,))
    c = np.zeros(10)
    _A = hcl.asarray(a)
    _B = hcl.asarray(b)
    _C = hcl.asarray(c, hcl.UInt(2))

    f(_A, _B, _C)

    _A = _A.asnumpy()
    _B = _B.asnumpy()
    _C = _C.asnumpy()

    for i in range(0, 10):
        assert(_C[i] == (a[i]+b[i]) % 4)

def test_module_quantize_ret_dtype():

    hcl.init()

    def algorithm(A, B):

        @hcl.module([A.shape, B.shape, ()])
        def add(A, B, x):
            hcl.return_(A[x] + B[x])

        return hcl.compute(A.shape, lambda x: add(A, B, x), "C")

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_scheme([A, B], algorithm)
    s.downsize([algorithm.add, algorithm.C], hcl.UInt(2))
    s = hcl.create_schedule_from_scheme(s)
    f = hcl.build(s)

    a = np.random.randint(100, size=(10,))
    b = np.random.randint(100, size=(10,))
    c = np.zeros(10)
    _A = hcl.asarray(a)
    _B = hcl.asarray(b)
    _C = hcl.asarray(c, hcl.UInt(2))

    f(_A, _B, _C)

    _A = _A.asnumpy()
    _B = _B.asnumpy()
    _C = _C.asnumpy()

    for i in range(0, 10):
        assert(_C[i] == (a[i]+b[i]) % 4)

def test_module_quantize_args():

    hcl.init()

    def algorithm(A, B):

        @hcl.module([A.shape, B.shape, ()])
        def add(A, B, x):
            hcl.return_(A[x] + B[x])

        return hcl.compute(A.shape, lambda x: add(A, B, x), "C")

    A = hcl.placeholder((10,), dtype=hcl.UInt(2))
    B = hcl.placeholder((10,))

    s = hcl.create_scheme([A, B], algorithm)
    s.downsize([algorithm.add.A], hcl.UInt(2))
    s = hcl.create_schedule_from_scheme(s)
    f = hcl.build(s)

    a = np.random.randint(100, size=(10,))
    b = np.random.randint(100, size=(10,))
    c = np.zeros(10)
    _A = hcl.asarray(a, hcl.UInt(2))
    _B = hcl.asarray(b)
    _C = hcl.asarray(c)

    f(_A, _B, _C)

    _A = _A.asnumpy()
    _B = _B.asnumpy()
    _C = _C.asnumpy()

    for i in range(0, 10):
        assert(_C[i] == a[i]%4 + b[i])
