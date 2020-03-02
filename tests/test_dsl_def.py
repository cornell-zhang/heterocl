import heterocl as hcl
import numpy as np

def test_module_no_return():

    def algorithm(A, B):

        @hcl.def_([A.shape, B.shape, ()])
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

        @hcl.def_([A.shape, ()])
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

def test_module_cond_return_if_only():

    def algorithm(A, B):

        @hcl.def_([A.shape, ()])
        def update_B(A, x):
            with hcl.if_(A[x] > 5):
                hcl.return_(-1)
            hcl.return_(A[x] + 1)

        hcl.update(B, lambda x: update_B(A, x))

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    a = np.random.randint(10, size=(10,))
    b = np.zeros(10)
    _A = hcl.asarray(a)
    _B = hcl.asarray(b, hcl.Int())

    f(_A, _B)

    _A = _A.asnumpy()
    _B = _B.asnumpy()

    for i in range(0, 10):
        assert(_B[i] == a[i]+1 if a[i] <=5 else -1)

def test_module_cond_return_if_else():

    def algorithm(A, B):

        @hcl.def_([A.shape, ()])
        def update_B(A, x):
            with hcl.if_(A[x] > 5):
                hcl.return_(-1)
            with hcl.else_():
                hcl.return_(A[x] + 1)

        hcl.update(B, lambda x: update_B(A, x))

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    a = np.random.randint(10, size=(10,))
    b = np.zeros(10)
    _A = hcl.asarray(a)
    _B = hcl.asarray(b, hcl.Int())

    f(_A, _B)

    _A = _A.asnumpy()
    _B = _B.asnumpy()

    for i in range(0, 10):
        assert(_B[i] == a[i]+1 if a[i] <=5 else -1)

def test_module_cond_return_multi_if_else():

    def algorithm(A, B):

        @hcl.def_([A.shape, ()])
        def update_B(A, x):
            with hcl.if_(A[x] > 5):
                with hcl.if_(A[x] > 7):
                    hcl.return_(-2)
                hcl.return_(-1)
            with hcl.else_():
                with hcl.if_(A[x] > 3):
                    hcl.return_(-3)
            hcl.return_(A[x] + 1)

        hcl.update(B, lambda x: update_B(A, x))

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    a = np.random.randint(10, size=(10,))
    b = np.zeros(10)
    _A = hcl.asarray(a)
    _B = hcl.asarray(b, hcl.Int())

    f(_A, _B)

    _A = _A.asnumpy()
    _B = _B.asnumpy()

    def check_res(val):
        if val > 5:
            if val > 7:
                return -2
            return -1
        else:
            if val > 3:
                return -3
        return val+1

    for i in range(0, 10):
        assert(_B[i] == check_res(a[i]))

def test_module_cond_return_for():

    def algorithm(A, B):

        @hcl.def_([A.shape, ()])
        def update_B(A, x):
            with hcl.for_(0, 10) as i:
                with hcl.if_(A[x] == i):
                    hcl.return_(1)
            hcl.return_(A[x])

        hcl.update(B, lambda x: update_B(A, x))

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    a = np.random.randint(20, size=(10,))
    b = np.zeros(10)
    _A = hcl.asarray(a)
    _B = hcl.asarray(b, hcl.Int())

    f(_A, _B)

    _A = _A.asnumpy()
    _B = _B.asnumpy()

    for i in range(0, 10):
        assert(_B[i] == 1 if a[i] < 10 else -1)

def test_module_multi_calls():

    def algorithm(A, B):

        @hcl.def_([A.shape, B.shape, ()])
        def add(A, B, x):
            hcl.return_(A[x] + B[x])

        @hcl.def_([A.shape, B.shape, ()])
        def mul(A, B, x):
            temp = hcl.scalar(0)
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

        @hcl.def_([A.shape, B.shape, ()], ret_dtype=hcl.UInt(2))
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

        @hcl.def_([A.shape, B.shape, ()])
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

def test_module_args_dtype():

    hcl.init()

    def algorithm(A, B):

        @hcl.def_([A.shape, B.shape, ()], [hcl.UInt(2), hcl.Int(32), hcl.Int(32)])
        def add(A, B, x):
            hcl.return_(A[x] + B[x])

        return hcl.compute(A.shape, lambda x: add(A, B, x), "C")

    A = hcl.placeholder((10,), dtype=hcl.UInt(2))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], algorithm)
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

def test_module_quantize_args():

    hcl.init()

    def algorithm(A, B):

        @hcl.def_([A.shape, B.shape, ()])
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

def test_module_declarative():
    hcl.init()

    def algorithm(a, b, c):

        @hcl.def_([a.shape, b.shape, c.shape])
        def add(a, b, c):
            hcl.update(c, lambda *x: a[x] + b[x])

        add(a, b, c)

    a = hcl.placeholder((10,))
    b = hcl.placeholder((10,))
    c = hcl.placeholder((10,))

    s = hcl.create_schedule([a, b, c], algorithm)
    f = hcl.build(s)

    a = np.random.randint(100, size=(10,))
    b = np.random.randint(100, size=(10,))
    c = np.zeros(10)
    _a = hcl.asarray(a)
    _b = hcl.asarray(b)
    _c = hcl.asarray(c)

    f(_a, _b, _c)

    assert np.array_equal(_c.asnumpy(), a + b)

def test_module_declarative_internal_allocate():
    hcl.init()

    def algorithm(a, b, c):

        @hcl.def_([a.shape, b.shape, c.shape])
        def add(a, b, c):
            d = hcl.compute(a.shape, lambda *x: a[x] + b[x])
            hcl.update(c, lambda *x: d[x] + 1)

        add(a, b, c)

    a = hcl.placeholder((10,))
    b = hcl.placeholder((10,))
    c = hcl.placeholder((10,))

    s = hcl.create_schedule([a, b, c], algorithm)
    f = hcl.build(s)

    a = np.random.randint(100, size=(10,))
    b = np.random.randint(100, size=(10,))
    c = np.zeros(10)
    _a = hcl.asarray(a)
    _b = hcl.asarray(b)
    _c = hcl.asarray(c)

    f(_a, _b, _c)

    assert np.array_equal(_c.asnumpy(), a + b + 1)

def test_module_declarative_compute_at():
    hcl.init()

    def algorithm(a, b, c):

        @hcl.def_([a.shape, b.shape, c.shape])
        def add(a, b, c):
            d = hcl.compute(a.shape, lambda *x: a[x] + b[x], "d")
            hcl.update(c, lambda *x: d[x] + 1, "u")

        add(a, b, c)

    a = hcl.placeholder((10,))
    b = hcl.placeholder((10,))
    c = hcl.placeholder((10,))

    s = hcl.create_schedule([a, b, c], algorithm)
    add = algorithm.add
    s[add.d].compute_at(s[add.u], add.u.axis[0])
    f = hcl.build(s)

    a = np.random.randint(100, size=(10,))
    b = np.random.randint(100, size=(10,))
    c = np.zeros(10)
    _a = hcl.asarray(a)
    _b = hcl.asarray(b)
    _c = hcl.asarray(c)

    f(_a, _b, _c)

    assert np.array_equal(_c.asnumpy(), a + b + 1)

def test_module_mixed_paradigm():
    hcl.init()

    def algorithm(a, b, c):

        @hcl.def_([a.shape, b.shape, c.shape])
        def add(a, b, c):
            with hcl.for_(0, 10) as i:
                a[i] = 0
            d = hcl.compute(a.shape, lambda *x: a[x] + b[x])
            hcl.update(c, lambda *x: d[x] + 1)

        add(a, b, c)

    a = hcl.placeholder((10,))
    b = hcl.placeholder((10,))
    c = hcl.placeholder((10,))

    s = hcl.create_schedule([a, b, c], algorithm)
    f = hcl.build(s)

    a = np.random.randint(100, size=(10,))
    b = np.random.randint(100, size=(10,))
    c = np.zeros(10)
    _a = hcl.asarray(a)
    _b = hcl.asarray(b)
    _c = hcl.asarray(c)

    f(_a, _b, _c)

    assert np.array_equal(_c.asnumpy(), b + 1)
