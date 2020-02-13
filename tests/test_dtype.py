import heterocl as hcl
import numpy as np

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

def test_dtype_long_int():
    # the longest we can support right now is 255-bit
    hcl.init(hcl.UInt(32))
    A = hcl.placeholder((100,))

    def kernel(A):
        B = hcl.compute(A.shape, lambda x: hcl.cast(hcl.UInt(255), A[x]) << 200, dtype=hcl.UInt(255))
        C = hcl.compute(A.shape, lambda x: B[x] >> 200)
        return C

    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)
    np_A = np.random.randint(0, 1<<31, 100)
    hcl_A = hcl.asarray(np_A)
    hcl_C = hcl.asarray(np.zeros(A.shape))
    f(hcl_A, hcl_C)

    assert np.array_equal(np_A, hcl_C.asnumpy())

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
        return E, F, G

    s = hcl.create_schedule([A, B, C], kernel)
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
