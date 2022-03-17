import heterocl as hcl
import numpy as np

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
        _B = hcl.asarray(np.zeros(40), dtype = hcl.UInt(i//4))

        f(_A, _B)

        __A = _A.asnumpy()
        __B = _B.asnumpy()

        for j in range(0, 10):
            for k in range(0, 4):
                numA = __A[j]
                numB = __B[j*4 + k]
                golden = (numA >> (i//4*k)) % (1 << (i//4))
                assert numB == golden

def test_unpack_dtype():

    def unpack(A, B):
        C = hcl.unpack(A, name = "C", dtype = B.dtype)
        hcl.update(B, lambda x: C[x])

    for i in range(4, 36, 4):
        A = hcl.placeholder((10,), "A", dtype = hcl.UInt(i))
        B = hcl.placeholder((40,), "B", dtype = hcl.UInt(i//4))

        s = hcl.create_schedule([A, B], unpack)
        f = hcl.build(s)

        _A = hcl.asarray(np.random.randint(1000, size = (10,)), dtype = hcl.UInt(i))
        _B = hcl.asarray(np.zeros(40), dtype = hcl.UInt(i//4))

        f(_A, _B)

        __A = _A.asnumpy()
        __B = _B.asnumpy()

        for j in range(0, 10):
            for k in range(0, 4):
                numA = __A[j]
                numB = __B[j*4 + k]
                golden = (numA >> (i//4*k)) % (1 << (i//4))
                assert numB == golden

def test_unpack_multi_dimension():

    def unpack(A):
        return hcl.unpack(A, axis=1, factor=4, name = "B")

    for i in range(4, 36, 4):
        A = hcl.placeholder((10, 10), "A", dtype = hcl.UInt(i))

        s = hcl.create_schedule([A], unpack)
        f = hcl.build(s)

        _A = hcl.asarray(np.random.randint(1000, size = (10, 10)), dtype = hcl.UInt(i))
        _B = hcl.asarray(np.zeros((10, 40)), dtype = hcl.UInt(i//4))

        f(_A, _B)

        __A = _A.asnumpy()
        __B = _B.asnumpy()

        for j in range(0, 10):
            for k in range(0, 10):
                for l in range(0, 4):
                    numA = __A[j, k]
                    numB = __B[j, k*4 + l]
                    golden = (numA >> (i//4*l)) % (1 << (i//4))
                    assert numB == golden

"""
Testing API: pack
"""
def test_pack():

    def pack(A):
        return hcl.pack(A, factor = 4)

    for i in range(4, 36, 4):
        A = hcl.placeholder((40,), "A", dtype = hcl.UInt(i//4))

        s = hcl.create_schedule([A], pack)
        f = hcl.build(s)

        _A = hcl.asarray(np.random.randint(1000, size = (40,)), dtype = hcl.UInt(i//4))
        _B = hcl.asarray(np.zeros(10), dtype = hcl.UInt(i))

        f(_A, _B)

        __A = _A.asnumpy()
        __B = _B.asnumpy()

        for j in range(0, 10):
            golden = 0
            numB = __B[j]
            for k in range(0, 4):
                numA = __A[j*4 + k]
                golden += numA << (k * i//4)
            assert numB == golden

def test_pack_dtype():

    def pack(A):
        return hcl.pack(A, dtype = hcl.UInt(A.type.bits*4))

    for i in range(4, 36, 4):
        A = hcl.placeholder((40,), "A", dtype = hcl.UInt(i//4))

        s = hcl.create_schedule([A], pack)
        f = hcl.build(s)

        _A = hcl.asarray(np.random.randint(1000, size = (40,)), dtype = hcl.UInt(i//4))
        _B = hcl.asarray(np.zeros(10), dtype = hcl.UInt(i))

        f(_A, _B)

        __A = _A.asnumpy()
        __B = _B.asnumpy()

        for j in range(0, 10):
            golden = 0
            numB = __B[j]
            for k in range(0, 4):
                numA = __A[j*4 + k]
                golden += numA << (k * i//4)
            assert numB == golden

def test_pack_multi_dimension():

    def pack(A):
        return hcl.pack(A, axis=1, factor=4)

    for i in range(4, 36, 4):
        A = hcl.placeholder((10, 40), "A", dtype = hcl.UInt(i//4))

        s = hcl.create_schedule([A], pack)
        f = hcl.build(s)

        _A = hcl.asarray(np.random.randint(1000, size = (10, 40)), dtype = hcl.UInt(i//4))
        _B = hcl.asarray(np.zeros((10, 10)), dtype = hcl.UInt(i))

        f(_A, _B)

        __A = _A.asnumpy()
        __B = _B.asnumpy()

        for j in range(0, 10):
            for k in range(0, 10):
                golden = 0
                numB = __B[j, k]
                for l in range(0, 4):
                    numA = __A[j, k*4 + l]
                    golden += numA << (l * i//4)
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


def test_pack_dtype_str():

    hcl.init(hcl.UInt(32))

    def kernel():
        A = hcl.compute((128,), lambda x: x, dtype="uint1")
        B = hcl.pack(A, dtype=hcl.UInt(32))
        return B

    s = hcl.create_schedule([], kernel)
    f = hcl.build(s)

    npB = np.zeros(4)
    hclB = hcl.asarray(npB, dtype=hcl.UInt(32))

    f(hclB)

    npB = hclB.asnumpy()

    for i in range(0, 4):
        e = npB[i]
        for j in range(0, 32):
            assert e%2 == j%2
            e = e >> 1
