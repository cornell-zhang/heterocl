import heterocl as hcl
import numpy as np
import pytest

def test_dtype_basic_uint():

    def _test_dtype(dtype):
        hcl.init(dtype)
        np_a = np.random.randint(0, 1<<63, 100)
        hcl_a = hcl.asarray(np_a)
        np_a2 = hcl_a.asnumpy()
        def cast(val):
            sb = 1 << dtype.bits
            val = val % sb
            return val
        vfunc = np.vectorize(cast)
        np_a3 = vfunc(np_a)
        assert np.array_equal(np_a2, np_a3)

    for j in range(0, 10):
        for i in range(2, 66, 4):
            _test_dtype(hcl.UInt(i))

def test_dtype_basic_int():

    def _test_dtype(dtype):
        hcl.init(dtype)
        s = 1 << 63
        np_a = np.random.randint(-s+1, s, 100)
        hcl_a = hcl.asarray(np_a)
        np_a2 = hcl_a.asnumpy()
        def cast(val):
            sb = 1 << dtype.bits
            sb1 = 1 << (dtype.bits-1)
            val = val % sb
            val = val if val < sb1 else val - sb
            return val
        vfunc = np.vectorize(cast)
        np_a3 = vfunc(np_a)
        assert np.array_equal(np_a2, np_a3)

    for j in range(0, 10):
        for i in range(2, 66, 4):
            _test_dtype(hcl.Int(i))

def test_dtype_basic_ufixed():

    def _test_dtype(dtype):
        hcl.init(dtype)
        np_A = np.random.rand(100)
        hcl_A = hcl.asarray(np_A)
        np_A2 = hcl_A.asnumpy()
        def cast(val):
            sf = 1 << dtype.fracs
            sb = 1 << dtype.bits
            val = val * sf
            val = int(val) % sb
            val = float(val) / sf
            return val
        vfunc = np.vectorize(cast)
        np_A3 = vfunc(np_A)
        assert np.array_equal(np_A2, np_A3)

    for j in range(0, 10):
        for i in range(2, 66, 4):
            _test_dtype(hcl.UFixed(i, i-2))

def test_dtype_overflow_ufixed():

    def _test_dtype(dtype):
        hcl.init(dtype)
        np_A = np.random.rand(100) * 10
        hcl_A = hcl.asarray(np_A)
        np_A2 = hcl_A.asnumpy()
        def cast(val):
            sf = 1 << dtype.fracs
            sb = 1 << dtype.bits
            val = val * sf
            val = int(val) % sb
            val = float(val) / sf
            return val
        vfunc = np.vectorize(cast)
        np_A3 = vfunc(np_A)
        assert np.array_equal(np_A2, np_A3)

    for j in range(0, 10):
        for i in range(2, 66, 4):
            _test_dtype(hcl.UFixed(i, i-2))

def test_dtype_basic_fixed():

    def _test_dtype(dtype):
        hcl.init(dtype)
        np_A = np.random.rand(100) - 0.5
        hcl_A = hcl.asarray(np_A)
        np_A2 = hcl_A.asnumpy()
        def cast(val):
            sf = 1 << dtype.fracs
            sb = 1 << dtype.bits
            sb1 = 1 << (dtype.bits-1)
            val = val * sf
            val = int(val) % sb
            val = val if val < sb1 else val - sb
            val = float(val) / sf
            return val
        vfunc = np.vectorize(cast)
        np_A3 = vfunc(np_A)
        assert np.array_equal(np_A2, np_A3)

    for j in range(0, 10):
        for i in range(2, 66, 4):
            _test_dtype(hcl.Fixed(i, i-2))

def test_dtype_overflow_fixed():

    def _test_dtype(dtype):
        hcl.init(dtype)
        np_A = (np.random.rand(100) - 0.5) * 100
        hcl_A = hcl.asarray(np_A)
        np_A2 = hcl_A.asnumpy()
        def cast(val):
            sf = 1 << dtype.fracs
            sb = 1 << dtype.bits
            sb1 = 1 << (dtype.bits-1)
            val = val * sf
            val = int(val) % sb
            val = val if val < sb1 else val - sb
            val = float(val) / sf
            return val
        vfunc = np.vectorize(cast)
        np_A3 = vfunc(np_A)
        assert np.array_equal(np_A2, np_A3)

    for j in range(0, 10):
        for i in range(2, 66, 4):
            _test_dtype(hcl.Fixed(i, i-2))

def test_dtype_basic_float():
    hcl.init(hcl.Float())
    np_A = np.random.rand(100) - 0.5
    hcl_A = hcl.asarray(np_A)
    np_A2 = hcl_A.asnumpy()
    assert np.allclose(np_A, np_A2)

def test_dtype_compute_fixed():

    def _test_dtype(dtype):
        hcl.init(dtype)
        A = hcl.placeholder((100,))
        B = hcl.placeholder((100,))

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda x: A[x] + B[x])
            D = hcl.compute(A.shape, lambda x: A[x] - B[x])
            E = hcl.compute(A.shape, lambda x: A[x] * B[x])
            # division is not recommended
            #F = hcl.compute(A.shape, lambda x: A[x] / B[x])
            #return C, D, E, F
            return C, D, E

        s = hcl.create_schedule([A, B], kernel)
        f = hcl.build(s)

        np_A = np.random.rand(*A.shape) + 0.1
        np_B = np.random.rand(*B.shape) + 0.1
        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B)
        hcl_C = hcl.asarray(np.zeros(A.shape))
        hcl_D = hcl.asarray(np.zeros(A.shape))
        hcl_E = hcl.asarray(np.zeros(A.shape))
        #hcl_F = hcl.asarray(np.zeros(A.shape))
        #f(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E, hcl_F)
        f(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E)

        np_C = hcl.cast_np(hcl_A.asnumpy() + hcl_B.asnumpy(), dtype)
        np_D = hcl.cast_np(hcl_A.asnumpy() - hcl_B.asnumpy(), dtype)
        np_E = hcl.cast_np(hcl_A.asnumpy() * hcl_B.asnumpy(), dtype)
        #np_F = hcl.cast_np(hcl_A.asnumpy() / hcl_B.asnumpy(), dtype)

        assert np.allclose(np_C, hcl_C.asnumpy())
        assert np.allclose(np_D, hcl_D.asnumpy())
        assert np.allclose(np_E, hcl_E.asnumpy())
        #assert np.allclose(np_F, hcl_F.asnumpy())

    for j in range(0, 10):
        for i in range(6, 66, 4):
            # To avoid floating point exception during division
            _test_dtype(hcl.UFixed(i, i-2))
            _test_dtype(hcl.Fixed(i, i-2))

def test_dtype_cast():

    def _test_body(dtype1, dtype2, dtype3):

        hcl.init()
        A = hcl.placeholder((100,), dtype=dtype1)
        B = hcl.placeholder((100,), dtype=dtype2)

        def kernel(A, B):
            C = hcl.compute((100,), lambda x: A[x] + B[x], dtype=dtype3)
            D = hcl.compute((100,), lambda x: A[x] - B[x], dtype=dtype3)
            return C, D

        s = hcl.create_schedule([A, B], kernel)
        f = hcl.build(s)

        npA = np.random.rand(100) * 100
        npB = np.random.rand(100) * 100
        npC = np.random.rand(100)
        npD = np.random.rand(100)

        hclA = hcl.asarray(npA, dtype1)
        hclB = hcl.asarray(npB, dtype2)
        hclC = hcl.asarray(npC, dtype3)
        hclD = hcl.asarray(npD, dtype3)

        f(hclA, hclB, hclC, hclD)

        # TODO: check results using HLS CSIM

    from itertools import permutations

    perm = permutations(
            [hcl.UInt(1), hcl.Int(1), hcl.UInt(10), hcl.Int(10),
             hcl.UInt(32), hcl.Int(32), hcl.UFixed(4, 2), hcl.Fixed(4, 2),
             hcl.UFixed(32, 16), hcl.Fixed(32, 16), hcl.Float()], 3)

    for dtypes in list(perm):
        _test_body(*dtypes)


def test_dtype_long_int():
    # the longest we can support right now is 2047-bit

    def test_kernel(bw, sl):
        hcl.init(hcl.UInt(32))
        A = hcl.placeholder((100,))
        B = hcl.placeholder((100,))

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda x: hcl.cast(hcl.UInt(bw), A[x]) << sl, dtype=hcl.UInt(bw))
            D = hcl.compute(A.shape, lambda x: B[x] + C[x], dtype=hcl.UInt(bw))
            E = hcl.compute(A.shape, lambda x: A[x])
            return E

        s = hcl.create_schedule([A, B], kernel)
        f = hcl.build(s)
        np_A = np.random.randint(0, 1<<31, 100)
        np_B = np.random.randint(0, 1<<31, 100)
        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B)
        hcl_E = hcl.asarray(np.zeros(A.shape))
        f(hcl_A, hcl_B, hcl_E)

        assert np.array_equal(np_A, hcl_E.asnumpy())

    test_kernel(64, 30)
    test_kernel(100, 60)
    test_kernel(250, 200)
    test_kernel(500, 400)
    test_kernel(1000, 750)
    test_kernel(2000, 1800)

def test_dtype_too_long_int():
    # the longest we can support right now is 2047-bit

    def test_kernel_total():
        A = hcl.placeholder((100,), dtype=hcl.Int(2048))

    def test_kernel_fracs():
        A = hcl.placeholder((100,), dtype=hcl.Fixed(1000, 800))

    for func in [test_kernel_total, test_kernel_fracs]:
        with pytest.raises(hcl.debug.DTypeError):
            func()

def test_dtype_struct():
    hcl.init()
    A = hcl.placeholder((100,), dtype=hcl.Int(8))
    B = hcl.placeholder((100,), dtype=hcl.Fixed(13, 11))
    C = hcl.placeholder((100,), dtype=hcl.Float())

    def kernel(A, B, C):
        stype = hcl.Struct({"fa": hcl.Int(8), "fb": hcl.Fixed(13, 11), "fc": hcl.Float()})
        D = hcl.compute(A.shape, lambda x: (A[x], B[x], C[x]), dtype=stype)
        E = hcl.compute(A.shape, lambda x: D[x].fa, dtype=hcl.Int(8))
        F = hcl.compute(A.shape, lambda x: D[x].fb, dtype=hcl.Fixed(13, 11))
        G = hcl.compute(A.shape, lambda x: D[x].fc, dtype=hcl.Float())
        # Check the data type
        assert D[0].fa.dtype == "int8"
        assert D[0].fb.dtype == "fixed13_11"
        assert D[0].fc.dtype == "float32"
        return E, F, G

    s = hcl.create_schedule([A, B, C], kernel)
    print(hcl.lower(s))
    f = hcl.build(s)
    np_A = np.random.randint(0, 500, size=100) - 250
    np_B = np.random.rand(100) - 0.5
    np_C = np.random.rand(100) - 0.5
    np_E = np.zeros(100)
    np_F = np.zeros(100)
    np_G = np.zeros(100)
    hcl_A = hcl.asarray(np_A, dtype=hcl.Int(8))
    hcl_B = hcl.asarray(np_B, dtype=hcl.Fixed(13, 11))
    hcl_C = hcl.asarray(np_C, dtype=hcl.Float())
    hcl_E = hcl.asarray(np_E, dtype=hcl.Int(8))
    hcl_F = hcl.asarray(np_F, dtype=hcl.Fixed(13, 11))
    hcl_G = hcl.asarray(np_G, dtype=hcl.Float())
    f(hcl_A, hcl_B, hcl_C, hcl_E, hcl_F, hcl_G)

    assert np.allclose(hcl_A.asnumpy(), hcl_E.asnumpy())
    assert np.allclose(hcl_B.asnumpy(), hcl_F.asnumpy())
    assert np.allclose(hcl_C.asnumpy(), hcl_G.asnumpy())

def test_dtye_strcut_complex():
    hcl.init()
    A = hcl.placeholder((100,))
    B = hcl.placeholder((100,))
    C = hcl.placeholder((100,))
    O = hcl.placeholder((100, 6))

    def kernel(A, B, C, O):
        dtype_xyz = hcl.Struct({"x": hcl.Int(), "y": hcl.Int(), "z": hcl.Int()})
        dtype_out = hcl.Struct({"v0": hcl.Int(),
                                "v1": hcl.Int(),
                                "v2": hcl.Int(),
                                "v3": hcl.Int(),
                                "v4": hcl.Int(),
                                "v5": hcl.Int()})

        D = hcl.compute(A.shape, lambda x: (A[x], B[x], C[x]), dtype=dtype_xyz)
        E = hcl.compute(A.shape, lambda x: (D[x].x * D[x].x,
                                            D[x].y * D[x].y,
                                            D[x].z * D[x].z,
                                            D[x].x * D[x].y,
                                            D[x].y * D[x].z,
                                            D[x].x * D[x].z), dtype=dtype_out)
        with hcl.Stage():
            with hcl.for_(0, 100) as i:
                for j in range(0, 6):
                    O[i][j] = E[i].__getattr__("v" + str(j))

    s = hcl.create_schedule([A, B, C, O], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=100)
    np_B = np.random.randint(10, size=100)
    np_C = np.random.randint(10, size=100)
    np_O = np.zeros((100, 6))

    np_G = np.zeros((100, 6)).astype("int")
    for i in range(0, 100):
        np_G[i][0] = np_A[i] * np_A[i]
        np_G[i][1] = np_B[i] * np_B[i]
        np_G[i][2] = np_C[i] * np_C[i]
        np_G[i][3] = np_A[i] * np_B[i]
        np_G[i][4] = np_B[i] * np_C[i]
        np_G[i][5] = np_A[i] * np_C[i]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_C = hcl.asarray(np_C)
    hcl_O = hcl.asarray(np_O)
    f(hcl_A, hcl_B, hcl_C, hcl_O)

    assert np.array_equal(hcl_O.asnumpy(), np_G)

def test_dtype_bit_slice():

    hcl.init(hcl.Int())

    def kernel():
        A = hcl.compute((10,), lambda x: x)
        assert A[0][0:4].dtype == "uint4"
        assert A[0][A[0]:A[4]].dtype == "int32"
        assert A[0][A[0]:A[0]+4].dtype == "uint4"
        return A

    s = hcl.create_schedule([], kernel)
    f = hcl.build(s)
    np_A = np.zeros((10,))
    hcl_A = hcl.asarray(np_A)
    f(hcl_A)

def test_dtype_const_long_int():

    hcl.init(hcl.Int())
    r = np.random.randint(0, 10, size=(1,))

    def kernel():
        A = hcl.compute((1,), lambda x: r[0], dtype=hcl.Int(128))
        B = hcl.compute((1,), lambda x: A[x])
        return B

    s = hcl.create_schedule([], kernel)
    f = hcl.build(s)
    np_B = np.zeros((1,))
    hcl_B = hcl.asarray(np_B)
    f(hcl_B)

    assert np.array_equal(r, hcl_B.asnumpy())

def test_dtype_large_array():

    def test_kernel(dtype):
        hcl.init(dtype)

        A = hcl.placeholder((1000,))

        def kernel(A):
            X = hcl.compute(A.shape, lambda x: A[x])
            return hcl.compute(A.shape, lambda x: X[x])

        s = hcl.create_schedule([A], kernel)
        f = hcl.build(s)

        npA = np.random.rand(1000)
        npB = np.zeros(1000)

        hcl_A = hcl.asarray(npA)
        hcl_B = hcl.asarray(npB)

        f(hcl_A, hcl_B)

        assert np.allclose(hcl_A.asnumpy(), hcl_B.asnumpy())

    test_kernel(hcl.Fixed(8, 6))
    test_kernel(hcl.Fixed(16, 14))
    test_kernel(hcl.Fixed(3, 1))
    test_kernel(hcl.Fixed(6, 4))
    test_kernel(hcl.Fixed(11, 9))
    test_kernel(hcl.Fixed(18, 16))
    test_kernel(hcl.Fixed(37, 35))

test_dtype_long_int()
