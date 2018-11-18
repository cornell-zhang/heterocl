import heterocl as hcl
import numpy as np

"""
Testing API: make schedule
"""

def test_base_update():
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

def test_base_return():
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

def test_base_return_multi():
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

def test_compute():
    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    def algorithm(A, B):
        C = hcl.compute(A.shape, lambda x: A[x] + 1)
        hcl.update(B, lambda x: C[x] + 1)

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    _A = hcl.asarray(np.random.randint(100, size=(10,)), dtype = hcl.Int(32))
    _B = hcl.asarray(np.zeros(10), dtype = hcl.Int(32))

    f(_A, _B)

    _A = _A.asnumpy()
    _B = _B.asnumpy()

    for i in range(10):
        assert(_B[i] == _A[i] + 2)

def test_schedule():
    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    def algorithm(A, B):
        hcl.update(B, lambda u0: A[u0] + 1, name="U")

    s = hcl.create_schedule([A, B], algorithm)
    s[algorithm.U].unroll(algorithm.U.axis[0])

    s = hcl.lower(s)

    assert 'unrolled "factor"=0 (u0, 0, 10)' in str(s)

"""
Testing API: make_scheme & make_schedule_from_scheme
"""

def test_resize():

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

"""
Testing API: module
"""
def test_module_no_return():

    hcl.init()

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
        assert(_B[i] == _A[i]+1)


def test_module_with_return():

    hcl.init()

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
        assert(_B[i] == _A[i]+1)

def test_module_multi_calls():

    hcl.init()

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
        assert(_C[i] == (_A[i]+_B[i])*i)

def test_module_ret_dtype():

    hcl.init()

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
        assert(_C[i] == (_A[i]+_B[i]) % 4)

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
        assert(_C[i] == (_A[i]+_B[i]) % 4)

"""
Testing API: unpack
"""
def test_unpack():

    def unpack(A):
        return hcl.unpack(A, factor = 4, name = "B")

    for i in range(4, 36, 4):
        A = hcl.placeholder((10,), "A", dtype = hcl.UInt(i))

        s = hcl.create_schedule([A], unpack)
        f = hcl.build(s)

        _A = hcl.asarray(np.random.randint(1000, size = (10,)), dtype = hcl.UInt(i))
        _B = hcl.asarray(np.zeros(40), dtype = hcl.UInt(i/4))

        f(_A, _B)

        __A = _A.asnumpy()
        __B = _B.asnumpy()

        for j in range(0, 10):
            for k in range(0, 4):
                numA = __A[j]
                numB = __B[j*4 + k]
                golden = (numA >> (i/4*k)) % (1 << (i/4))
                assert numB == golden

def test_unpack_dtype():

    def unpack(A, B):
        C = hcl.unpack(A, name = "C", dtype = B.dtype)
        hcl.update(B, lambda x: C[x])

    for i in range(4, 36, 4):
        A = hcl.placeholder((10,), "A", dtype = hcl.UInt(i))
        B = hcl.placeholder((40,), "B", dtype = hcl.UInt(i/4))

        s = hcl.create_schedule([A, B], unpack)
        f = hcl.build(s)

        _A = hcl.asarray(np.random.randint(1000, size = (10,)), dtype = hcl.UInt(i))
        _B = hcl.asarray(np.zeros(40), dtype = hcl.UInt(i/4))

        f(_A, _B)

        __A = _A.asnumpy()
        __B = _B.asnumpy()

        for j in range(0, 10):
            for k in range(0, 4):
                numA = __A[j]
                numB = __B[j*4 + k]
                golden = (numA >> (i/4*k)) % (1 << (i/4))
                assert numB == golden

"""
Testing API: pack
"""
def test_pack():

    def pack(A):
        return hcl.pack(A, factor = 4)

    for i in range(4, 36, 4):
        A = hcl.placeholder((40,), "A", dtype = hcl.UInt(i/4))

        s = hcl.create_schedule([A], pack)
        f = hcl.build(s)

        _A = hcl.asarray(np.random.randint(1000, size = (40,)), dtype = hcl.UInt(i/4))
        _B = hcl.asarray(np.zeros(10), dtype = hcl.UInt(i))

        f(_A, _B)

        __A = _A.asnumpy()
        __B = _B.asnumpy()

        for j in range(0, 10):
            golden = 0
            numB = __B[j]
            for k in range(0, 4):
                numA = __A[j*4 + k]
                golden += numA << (k * i/4)
            assert numB == golden

def test_pack_dtype():

    def pack(A):
        return hcl.pack(A, dtype = hcl.UInt(A.type.bits*4))

    for i in range(4, 36, 4):
        A = hcl.placeholder((40,), "A", dtype = hcl.UInt(i/4))

        s = hcl.create_schedule([A], pack)
        f = hcl.build(s)

        _A = hcl.asarray(np.random.randint(1000, size = (40,)), dtype = hcl.UInt(i/4))
        _B = hcl.asarray(np.zeros(10), dtype = hcl.UInt(i))

        f(_A, _B)

        __A = _A.asnumpy()
        __B = _B.asnumpy()

        for j in range(0, 10):
            golden = 0
            numB = __B[j]
            for k in range(0, 4):
                numA = __A[j*4 + k]
                golden += numA << (k * i/4)
            assert numB == golden

def test_pack_unpack():

    def pack_unpack(A):
        C = hcl.pack(A, factor = 4)
        return hcl.unpack(C, factor = 4)

    for i in range(1, 16):
        A = hcl.placeholder((40,), "A", dtype = hcl.UInt(i))

        s = hcl.create_schedule([A], pack_unpack)
        f = hcl.build(s)

        _A = hcl.asarray(np.random.randint(1000, size = (40,)), dtype = hcl.UInt(i))
        _B = hcl.asarray(np.zeros(40), dtype = hcl.UInt(i))

        f(_A, _B)

        __A = _A.asnumpy()
        __B = _B.asnumpy()

        for j in range(0, 40):
            assert __A[j] == __B[j]
