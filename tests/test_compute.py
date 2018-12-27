import heterocl as hcl
import numpy

def _test_kernel(kernel):
    A = hcl.placeholder((10,))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = numpy.random.randint(10, size=(10,))
    np_B = numpy.zeros(10)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()

    for i in range(0, 10):
        assert ret_B[i] == np_A[i]+1

def test_fcompute_basic():

    def kernel(A):
        return hcl.compute(A.shape, lambda x: A[x]+1)

    _test_kernel(kernel)

def test_fcompute_function_wrapper():

    def kernel(A):
        def foo(x):
            return x+1
        return hcl.compute(A.shape, lambda x: foo(A[x]))

    _test_kernel(kernel)

def test_fcompute_wrap_more():

    def kernel(A):
        def foo(x):
            return A[x]+1
        return hcl.compute(A.shape, lambda x: foo(x))

    _test_kernel(kernel)

def test_fcompute_no_lambda():

    def kernel(A):
        def foo(x):
            return A[x]+1
        return hcl.compute(A.shape, foo)

    _test_kernel(kernel)

def test_fcompute_imperative_return():

    def kernel(A):
        def foo(x):
            hcl.return_(A[x]+1)
        return hcl.compute(A.shape, foo)

    _test_kernel(kernel)

def test_fcompute_imperative_function():

    def kernel(A):
        @hcl.module([A.shape, ()])
        def foo(A, x):
            hcl.return_(A[x]+1)
        return hcl.compute(A.shape, lambda x: foo(A, x))

    _test_kernel(kernel)

def test_fcompute_multiple_return():

    def kernel(A):
        def foo(x):
            with hcl.if_(A[x] > 5):
                hcl.return_(x)
            with hcl.else_():
                hcl.return_(0)
        return hcl.compute(A.shape, foo)

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = numpy.random.randint(10, size=(10,))
    np_B = numpy.zeros(10)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()

    for i in range(0, 10):
        if np_A[i] > 5:
            assert ret_B[i] == i
        else:
            assert ret_B[i] == 0
