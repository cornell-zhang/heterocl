import heterocl as hcl
import numpy as np

def test_basic(vhls):
    if not vhls:
        return
    hcl.init()
    A = hcl.placeholder((10,))
    def kernel(A):
        return hcl.compute(A.shape, lambda x: A[x] + 1)
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s, target='vhls_csim')
    np_A = np.random.randint(0, 10, A.shape)
    np_B = np.zeros(A.shape)
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    f(hcl_A, hcl_B)
    np_B = np_A + 1
    np.testing.assert_array_equal(np_B, hcl_B.asnumpy())

def test_floating_point(vhls):
    if not vhls:
        return
    hcl.init(hcl.Float())
    A = hcl.placeholder((10,))
    def kernel(A):
        return hcl.compute(A.shape, lambda x: A[x] + 1)
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s, target='vhls_csim')
    np_A = np.random.rand(*A.shape)
    np_B = np.zeros(A.shape)
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    f(hcl_A, hcl_B)
    np_B = np_A + 1
    np.testing.assert_allclose(np_B, hcl_B.asnumpy())

def test_ap_uint(vhls):
    if not vhls:
        return
    for i in range (2, 40, 4):
        hcl.init(hcl.UInt(i))
        A = hcl.placeholder((10,))
        def kernel(A):
            return hcl.compute(A.shape, lambda x: A[x] + 1)
        s = hcl.create_schedule(A, kernel)
        f = hcl.build(s, target='vhls_csim')
        f2 = hcl.build(s)
        np_A = np.random.randint(0, 1<<40, A.shape)
        np_B = np.zeros(A.shape)
        hcl_A1 = hcl.asarray(np_A)
        hcl_B1 = hcl.asarray(np_B)
        hcl_A2 = hcl.asarray(np_A)
        hcl_B2 = hcl.asarray(np_B)
        f(hcl_A1, hcl_B1)
        f2(hcl_A2, hcl_B2)
        np.testing.assert_array_equal(hcl_B1.asnumpy(), hcl_B2.asnumpy())

def test_ap_int(vhls):
    if not vhls:
        return
    for i in range (2, 40, 4):
        hcl.init(hcl.UInt(i))
        A = hcl.placeholder((10,))
        def kernel(A):
            return hcl.compute(A.shape, lambda x: A[x] + 1)
        s = hcl.create_schedule(A, kernel)
        f = hcl.build(s, target='vhls_csim')
        f2 = hcl.build(s)
        np_A = np.random.randint(0, 1<<40, A.shape)
        np_B = np.zeros(A.shape)
        hcl_A1 = hcl.asarray(np_A)
        hcl_B1 = hcl.asarray(np_B)
        hcl_A2 = hcl.asarray(np_A)
        hcl_B2 = hcl.asarray(np_B)
        f(hcl_A1, hcl_B1)
        f2(hcl_A2, hcl_B2)
        np.testing.assert_array_equal(hcl_B1.asnumpy(), hcl_B2.asnumpy())

def test_ap_ufixed(vhls):
    if not vhls:
        return
    for i in range (2, 40, 4):
        hcl.init(hcl.UFixed(i, i-2))
        A = hcl.placeholder((10,))
        def kernel(A):
            return hcl.compute(A.shape, lambda x: A[x] + 1)
        s = hcl.create_schedule(A, kernel)
        f = hcl.build(s, target='vhls_csim')
        f2 = hcl.build(s)
        np_A = np.random.rand(*A.shape)
        np_B = np.zeros(A.shape)
        hcl_A1 = hcl.asarray(np_A)
        hcl_B1 = hcl.asarray(np_B)
        hcl_A2 = hcl.asarray(np_A)
        hcl_B2 = hcl.asarray(np_B)
        f(hcl_A1, hcl_B1)
        f2(hcl_A2, hcl_B2)
        np.testing.assert_allclose(hcl_B1.asnumpy(), hcl_B2.asnumpy())

def test_ap_fixed(vhls):
    if not vhls:
        return
    for i in range (2, 40, 4):
        hcl.init(hcl.Fixed(i, i-2))
        A = hcl.placeholder((10,))
        def kernel(A):
            return hcl.compute(A.shape, lambda x: A[x] + 1)
        s = hcl.create_schedule(A, kernel)
        f = hcl.build(s, target='vhls_csim')
        f2 = hcl.build(s)
        np_A = np.random.rand(*A.shape)
        np_B = np.zeros(A.shape)
        hcl_A1 = hcl.asarray(np_A)
        hcl_B1 = hcl.asarray(np_B)
        hcl_A2 = hcl.asarray(np_A)
        hcl_B2 = hcl.asarray(np_B)
        f(hcl_A1, hcl_B1)
        f2(hcl_A2, hcl_B2)
        np.testing.assert_allclose(hcl_B1.asnumpy(), hcl_B2.asnumpy())

def test_allocate(vhls):
    if not vhls:
        return
    hcl.init()
    A = hcl.placeholder((10,))
    def kernel(A):
        B = hcl.compute(A.shape, lambda x: A[x] + 1)
        return hcl.compute(A.shape, lambda x: B[x] + 1)
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s, target='vhls_csim')
    f2 = hcl.build(s)
    np_A = np.random.randint(0, 10, A.shape)
    np_C = np.zeros(A.shape)
    hcl_A1 = hcl.asarray(np_A)
    hcl_C1 = hcl.asarray(np_C)
    hcl_A2 = hcl.asarray(np_A)
    hcl_C2 = hcl.asarray(np_C)
    f(hcl_A1, hcl_C1)
    f2(hcl_A2, hcl_C2)
    np.testing.assert_array_equal(hcl_C1.asnumpy(), hcl_C2.asnumpy())

def test_multi_dim(vhls):
    if not vhls:
        return
    hcl.init()
    A = hcl.placeholder((5, 8))
    def kernel(A):
        B = hcl.compute(A.shape, lambda x, y: A[x, y] + 1)
        return hcl.compute(A.shape, lambda x, y: B[x, y] + 1)
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s, target='vhls_csim')
    f2 = hcl.build(s)
    np_A = np.random.randint(0, 10, A.shape)
    np_C = np.zeros(A.shape)
    hcl_A1 = hcl.asarray(np_A)
    hcl_C1 = hcl.asarray(np_C)
    hcl_A2 = hcl.asarray(np_A)
    hcl_C2 = hcl.asarray(np_C)
    f(hcl_A1, hcl_C1)
    f2(hcl_A2, hcl_C2)
    np.testing.assert_array_equal(hcl_C1.asnumpy(), hcl_C2.asnumpy())

def test_multi_dim2(vhls):
    if not vhls:
        return
    hcl.init()
    A = hcl.placeholder((5, 8, 12))
    def kernel(A):
        B = hcl.compute(A.shape, lambda x, y, z: A[x, y, z] + 1)
        return hcl.compute(A.shape, lambda x, y, z: B[x, y, z] + 1)
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s, target='vhls_csim')
    f2 = hcl.build(s)
    np_A = np.random.randint(0, 10, A.shape)
    np_C = np.zeros(A.shape)
    hcl_A1 = hcl.asarray(np_A)
    hcl_C1 = hcl.asarray(np_C)
    hcl_A2 = hcl.asarray(np_A)
    hcl_C2 = hcl.asarray(np_C)
    f(hcl_A1, hcl_C1)
    f2(hcl_A2, hcl_C2)
    np.testing.assert_array_equal(hcl_C1.asnumpy(), hcl_C2.asnumpy())

def test_get_bit(vhls):
    if not vhls:
        return
    hcl.init()
    A = hcl.placeholder((10,))
    def kernel(A):
        return hcl.compute(A.shape, lambda x: A[x][0])
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s, target='vhls_csim')
    f2 = hcl.build(s)
    np_A = np.random.randint(0, 10, A.shape)
    np_B = np.zeros(A.shape)
    hcl_A1 = hcl.asarray(np_A)
    hcl_B1 = hcl.asarray(np_B)
    hcl_A2 = hcl.asarray(np_A)
    hcl_B2 = hcl.asarray(np_B)
    f(hcl_A1, hcl_B1)
    f2(hcl_A2, hcl_B2)
    np.testing.assert_array_equal(hcl_B1.asnumpy(), hcl_B2.asnumpy())

def test_get_slice(vhls):
    if not vhls:
        return
    hcl.init()
    A = hcl.placeholder((10,))
    def kernel(A):
        return hcl.compute(A.shape, lambda x: A[x][5:0])
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s, target='vhls_csim')
    f2 = hcl.build(s)
    np_A = np.random.randint(0, 10, A.shape)
    np_B = np.zeros(A.shape)
    hcl_A1 = hcl.asarray(np_A)
    hcl_B1 = hcl.asarray(np_B)
    hcl_A2 = hcl.asarray(np_A)
    hcl_B2 = hcl.asarray(np_B)
    f(hcl_A1, hcl_B1)
    f2(hcl_A2, hcl_B2)
    np.testing.assert_array_equal(hcl_B1.asnumpy(), hcl_B2.asnumpy())

def test_set_bit(vhls):
    if not vhls:
        return
    hcl.init()
    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))
    def kernel(A, B):
        with hcl.Stage():
            with hcl.for_(0, 10) as i:
                B[i][0] = A[i][0]
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s, target='vhls_csim')
    f2 = hcl.build(s)
    np_A = np.random.randint(0, 10, A.shape)
    np_B = np.zeros(A.shape)
    hcl_A1 = hcl.asarray(np_A)
    hcl_B1 = hcl.asarray(np_B)
    hcl_A2 = hcl.asarray(np_A)
    hcl_B2 = hcl.asarray(np_B)
    f(hcl_A1, hcl_B1)
    f2(hcl_A2, hcl_B2)
    np.testing.assert_array_equal(hcl_B1.asnumpy(), hcl_B2.asnumpy())

def test_set_slice(vhls):
    if not vhls:
        return
    hcl.init()
    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))
    def kernel(A, B):
        with hcl.Stage():
            with hcl.for_(0, 10) as i:
                B[i][5:0] = A[i][5:0]
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s, target='vhls_csim')
    f2 = hcl.build(s)
    np_A = np.random.randint(0, 10, A.shape)
    np_B = np.zeros(A.shape)
    hcl_A1 = hcl.asarray(np_A)
    hcl_B1 = hcl.asarray(np_B)
    hcl_A2 = hcl.asarray(np_A)
    hcl_B2 = hcl.asarray(np_B)
    f(hcl_A1, hcl_B1)
    f2(hcl_A2, hcl_B2)
    np.testing.assert_array_equal(hcl_B1.asnumpy(), hcl_B2.asnumpy())
