import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np

def test_two_stages():
    hcl.init()
    A = hcl.placeholder((10,), "A")
    B = hcl.placeholder((10,), "B")
    C = hcl.placeholder((10,), "C")

    def kernel(A, B, C):

        @hcl.def_([A.shape, B.shape])
        def M1(A, B):
            with hcl.for_(0, 10) as i:
                B[i] = A[i] + 1

        @hcl.def_([B.shape, C.shape])
        def M2(B, C):
            with hcl.for_(0, 10) as i:
                C[i] = B[i] + 1

        M1(A, B)
        M2(B, C)

    s = hcl.create_schedule([A, B, C], kernel)
    s.to(B, s[kernel.M2], s[kernel.M1], depth=1)
    f = hcl.build(s)

    a = np.random.randint(100, size=(10,))
    b = np.random.randint(100, size=(10,))
    c = np.random.randint(100, size=(10,))
    hcl_A = hcl.asarray(a)
    hcl_B = hcl.asarray(b)
    hcl_C = hcl.asarray(c)

    f(hcl_A, hcl_B, hcl_C)
    np.testing.assert_array_equal(hcl_C.asnumpy(), a + 2)

def test_three_stages():
    hcl.init()
    A = hcl.placeholder((10,), "A")
    B = hcl.placeholder((10,), "B")
    C = hcl.placeholder((10,), "C")
    D = hcl.placeholder((10,), "D")

    def kernel(A, B, C, D):

        @hcl.def_([A.shape, B.shape])
        def M1(A, B):
            with hcl.for_(0, 10) as i:
                B[i] = A[i] + 1

        @hcl.def_([B.shape, C.shape])
        def M2(B, C):
            with hcl.for_(0, 10) as i:
                C[i] = B[i] + 1

        @hcl.def_([C.shape, D.shape])
        def M3(C, D):
            with hcl.for_(0, 10) as i:
                D[i] = C[i] + 1

        M1(A, B)
        M2(B, C)
        M3(C, D)

    s = hcl.create_schedule([A, B, C, D], kernel)
    s.to(B, s[kernel.M2], s[kernel.M1], depth=1)
    s.to(C, s[kernel.M3], s[kernel.M2], depth=1)
    f = hcl.build(s)

    a = np.random.randint(100, size=(10,))
    b = np.random.randint(100, size=(10,))
    c = np.random.randint(100, size=(10,))
    d = np.random.randint(100, size=(10,))
    hcl_A = hcl.asarray(a)
    hcl_B = hcl.asarray(b)
    hcl_C = hcl.asarray(c)
    hcl_D = hcl.asarray(d)

    f(hcl_A, hcl_B, hcl_C, hcl_D)
    np.testing.assert_array_equal(hcl_D.asnumpy(), a + 3)

def test_internal_stages():
    hcl.init()
    A = hcl.placeholder((10,), "A")
    B = hcl.placeholder((10,), "B")
    C = hcl.placeholder((10,), "C")
    D = hcl.placeholder((10,), "D")

    def kernel(A, B, C, D):

        @hcl.def_([A.shape, B.shape, C.shape, D.shape])
        def M1(A, B, C, D):
            with hcl.for_(0, 10) as i:
                B[i] = A[i] + 1
                D[i] = C[i] + 1

        @hcl.def_([B.shape, C.shape])
        def M2(B, C):
            with hcl.for_(0, 10) as i:
                C[i] = B[i] + 1

        M1(A, B, C, D)
        M2(B, C)

    s = hcl.create_schedule([A, B, C, D], kernel)
    s.to(B, s[kernel.M2], s[kernel.M1], depth=1)
    s.to(C, s[kernel.M1], s[kernel.M2], depth=1)
    f = hcl.build(s)

    a = np.random.randint(100, size=(10,))
    b = np.random.randint(100, size=(10,))
    c = np.random.randint(100, size=(10,))
    d = np.random.randint(100, size=(10,))
    hcl_A = hcl.asarray(a)
    hcl_B = hcl.asarray(b)
    hcl_C = hcl.asarray(c)
    hcl_D = hcl.asarray(d)

    f(hcl_A, hcl_B, hcl_C, hcl_D)
    np.testing.assert_array_equal(hcl_D.asnumpy(), a + 3)

def test_fork_stages():
    hcl.init()
    A = hcl.placeholder((10,), "A")
    B = hcl.placeholder((10,), "B")
    C = hcl.placeholder((10,), "C")
    D = hcl.placeholder((10,), "D")
    E = hcl.placeholder((10,), "E")

    def kernel(A, B, C, D, E):

        @hcl.def_([A.shape, B.shape, C.shape])
        def M1(A, B, C):
            with hcl.for_(0, 10) as i:
                B[i] = A[i] + 1
                C[i] = A[i] - 1

        @hcl.def_([B.shape, D.shape])
        def M2(B, D):
            with hcl.for_(0, 10) as i:
                D[i] = B[i] + 1

        @hcl.def_([C.shape, E.shape])
        def M3(C, E):
            with hcl.for_(0, 10) as i:
                E[i] = C[i] - 1

        M1(A, B, C)
        M2(B, D)
        M3(C, E)

    hcl.init()
    s1 = hcl.create_schedule([A, B, C, D, E], kernel)
    s1.to(B, s1[kernel.M2], s1[kernel.M1], depth=1)

    hcl.init()
    s2 = hcl.create_schedule([A, B, C, D, E], kernel)
    s2.to(C, s2[kernel.M3], s2[kernel.M1], depth=1)

    hcl.init()
    s3 = hcl.create_schedule([A, B, C, D, E], kernel)
    s3.to(B, s3[kernel.M2], s3[kernel.M1], depth=1)
    s3.to(C, s3[kernel.M3], s3[kernel.M1], depth=1)

    a = np.random.randint(100, size=(10,))
    b = np.random.randint(100, size=(10,))
    c = np.random.randint(100, size=(10,))
    d = np.random.randint(100, size=(10,))
    e = np.random.randint(100, size=(10,))

    def _test_stream(s):
        f = hcl.build(s)

        hcl_A = hcl.asarray(a)
        hcl_B = hcl.asarray(b)
        hcl_C = hcl.asarray(c)
        hcl_D = hcl.asarray(d)
        hcl_E = hcl.asarray(e)

        f(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E)
        np.testing.assert_array_equal(hcl_D.asnumpy(), a + 2)

    _test_stream(s1)
    _test_stream(s2)
    _test_stream(s3)

def test_merge_stages():
    hcl.init()
    A = hcl.placeholder((10,), "A")
    B = hcl.placeholder((10,), "B")
    C = hcl.placeholder((10,), "C")
    D = hcl.placeholder((10,), "D")
    E = hcl.placeholder((10,), "E")

    def kernel(A, B, C, D, E):

        @hcl.def_([A.shape, B.shape])
        def M1(A, B):
            with hcl.for_(0, 10) as i:
                B[i] = A[i] + 1

        @hcl.def_([C.shape, D.shape])
        def M2(C, D):
            with hcl.for_(0, 10) as i:
                D[i] = C[i] - 1

        @hcl.def_([B.shape, D.shape, E.shape])
        def M3(B, D, E):
            with hcl.for_(0, 10) as i:
                E[i] = B[i] + D[i]

        M1(A, B)
        M2(C, D)
        M3(B, D, E)

    hcl.init()
    s1 = hcl.create_schedule([A, B, C, D, E], kernel)
    s1.to(B, s1[kernel.M3], s1[kernel.M1], depth=1)

    hcl.init()
    s2 = hcl.create_schedule([A, B, C, D, E], kernel)
    s2.to(D, s2[kernel.M3], s2[kernel.M2], depth=1)

    hcl.init()
    s3 = hcl.create_schedule([A, B, C, D, E], kernel)
    #s3.to(B, s3[kernel.M3], s3[kernel.M1], depth=1)
    #s3.to(D, s3[kernel.M3], s3[kernel.M2], depth=1)
    print(hcl.lower(s3))

    a = np.random.randint(100, size=(10,))
    b = np.random.randint(100, size=(10,))
    c = np.random.randint(100, size=(10,))
    d = np.random.randint(100, size=(10,))
    e = np.random.randint(100, size=(10,))

    def _test_stream(s):
        f = hcl.build(s)

        hcl_A = hcl.asarray(a)
        hcl_B = hcl.asarray(b)
        hcl_C = hcl.asarray(c)
        hcl_D = hcl.asarray(d)
        hcl_E = hcl.asarray(e)

        f(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E)
        np.testing.assert_array_equal(hcl_E.asnumpy(), a + c)

    #_test_stream(s1)
    #_test_stream(s2)
    _test_stream(s3)

def test_loop_stages():
    hcl.init()
    A = hcl.placeholder((10,), "A")
    B = hcl.placeholder((10,), "B")

    def kernel(A, B):

        @hcl.def_([A.shape, B.shape])
        def M1(A, B):
            with hcl.for_(0, 10) as i:
                with hcl.for_(0, 10) as j:
                    with hcl.if_(i == 0):
                        B[j] = A[j]
                    with hcl.else_():
                        B[j] = B[j] + 1

        M1(A, B)

    s = hcl.create_schedule([A, B], kernel)
    #s.to(B, s[kernel.M1], s[kernel.M1], depth=1)
    f = hcl.build(s)

    a = np.random.randint(100, size=(10,))
    b = np.random.randint(100, size=(10,))
    hcl_A = hcl.asarray(a)
    hcl_B = hcl.asarray(b)

    f(hcl_A, hcl_B)
    np.testing.assert_array_equal(hcl_B.asnumpy(), a + 9)

test_loop_stages()
