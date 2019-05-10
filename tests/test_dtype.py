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
