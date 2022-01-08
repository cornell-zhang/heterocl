import heterocl as hcl
import heterocl.tvm as tvm
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

def test_bitwise_and():
    hcl.init(hcl.UInt(8))

    N = 100
    A = hcl.placeholder((N, N))
    B = hcl.placeholder((N, N))

    def kernel(A, B):
        return hcl.compute(A.shape, lambda x, y: A[x, y] & B[x, y])

    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    a = np.random.randint(0, 255, (N, N))
    b = np.random.randint(0, 255, (N, N))
    c = np.zeros((N, N))
    g = a & b

    hcl_a = hcl.asarray(a)
    hcl_b = hcl.asarray(b)
    hcl_c = hcl.asarray(c)
    f(hcl_a, hcl_b, hcl_c)
    assert np.array_equal(hcl_c.asnumpy(), g)

def test_bitwise_or():
    hcl.init(hcl.UInt(8))

    N = 100
    A = hcl.placeholder((N, N))
    B = hcl.placeholder((N, N))

    def kernel(A, B):
        return hcl.compute(A.shape, lambda x, y: A[x, y] | B[x, y])

    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    a = np.random.randint(0, 255, (N, N))
    b = np.random.randint(0, 255, (N, N))
    c = np.zeros((N, N))
    g = a | b

    hcl_a = hcl.asarray(a)
    hcl_b = hcl.asarray(b)
    hcl_c = hcl.asarray(c)
    f(hcl_a, hcl_b, hcl_c)
    assert np.array_equal(hcl_c.asnumpy(), g)

def test_tesnro_slice_shape():
    A = hcl.placeholder((3, 4, 5))

    assert(A.shape == (3, 4, 5))
    assert(A[0].shape == (4, 5))
    assert(A[0][1].shape == (5,))

def test_build_from_stmt():
    hcl.init(hcl.Int())
    # First, we still need to create HeteroCL inputs
    A = hcl.placeholder((10,), "A")
    B = hcl.placeholder((10,), "B")
    X = hcl.placeholder((), "X") # a scalar input

    # Second, we create variables for loop var
    # The first field is the name
    # The second field is the data type
    i = tvm._api_internal._Var("i", "int32")

    # Similarly, we can create a variable for intermediate tensor
    C = tvm._api_internal._Var("C", "int32")

    # Third, we can create Load
    # If we are accessing the HeteroCL inputs, we need to use ".buf.data"
    load = tvm.make.Load("int32", A.buf.data, i)

    # Fourth, for arithmatic operation, we can add "False" to the end
    # This avoids automatic casting
    add = tvm.make.Add(load, 1, False)

    # Fifth, we can create Store
    # In this case, we just write to the intermediate tensor
    # Thus, we don't need to use ".buf.data"
    store = tvm.make.Store(C, add, i)

    # Sixth, we can create the loop with our loop var
    # For the details of each field, please refer to IR.h under HalideIR/src/ir
    loop = tvm.make.For(i, 0, 10, 0, 0, store)

    # Finally, we need to allocate memory for our intermediate tensor
    alloc = tvm.make.Allocate(C, "int32", [10], tvm.const(1, "uint1"), loop, [])

    # Similarly, we can do another loop that write stuffs to B
    # Note that this i is a newly allocated variable though the name is the same
    # We cannot reuse the same i for different loops
    i = tvm._api_internal._Var("i", "int32")
    load = tvm.make.Load("int32", C, i)
    mul = tvm.make.Mul(load, X, False)
    store = tvm.make.Store(B.buf.data, mul, i)
    loop = tvm.make.For(i, 0, 10, 0, 0, store)
    stmt = tvm.make.Block(alloc, loop)

    # Finally, we just need to use HeteroCL APIs to build the function
    # Note that with this approach, we cannot apply any optimizations with primitives
    s = hcl.create_schedule([A, B, X])
    # Just specify the stmt to be the statement we built
    f = hcl.build(s, stmt=stmt)

    # A simple test
    np_A = np.random.randint(10, size=10)
    np_B = np.random.randint(10, size=10)
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B, 5)

    np_golden = 5 * (np_A + 1)
    np_B = hcl_B.asnumpy()

    assert(np.array_equal(np_B, np_golden))
