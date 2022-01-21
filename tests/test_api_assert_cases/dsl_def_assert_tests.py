import heterocl as hcl
import numpy as np
m = 64
k = 64

def test_assert_module_no_return():
    hcl.init(raise_assert_exception=False)

    def algorithm(A, B):

        @hcl.def_([A.shape, B.shape, ()])
        def update_B(A, B, x):
            hcl.print(0, "print1\n")
            hcl.assert_(x < 2, "assert error 1")
            hcl.print(0, "print2\n")
            hcl.assert_(x < 1, "assert error 2")
            hcl.print(x, "print3\n")
            B[x] = A[x] + 1

        with hcl.Stage():
            matrix_B = hcl.compute((m,k), lambda x, y: A[x] + B[x] + 1, "matrix_B")
            with hcl.for_(0, 10) as i:
                matrix_C = hcl.compute((m,k), lambda x, y: A[x] + B[x] + 2, "matrix_C")
                with hcl.for_(0, 10) as z:
                    matrix_D = hcl.compute((m,k), lambda x, y: A[x] + B[x] + 3, "matrix_D")
                    update_B(A, B, i)
                    matrix_E = hcl.compute((m,k), lambda x, y: A[x] + B[x] + 4, "matrix_E")
            hcl.print(0, "end\n")

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    a = np.random.randint(100, size=(10,))
    b = np.zeros(10)
    _A = hcl.asarray(a)
    _B = hcl.asarray(b)

    # assert error 2 condition becomes false on 11th iteration of update_B
    # only print1 and print2 should print on the 11th iteration
    f(_A, _B)

def test_assert_module_with_return():
    hcl.init(raise_assert_exception=False)

    def algorithm(A, B):

        @hcl.def_([A.shape, ()])
        def update_B(A, x):
            hcl.print(0, "print1\n")
            hcl.assert_(A[x] != 7)
            hcl.print(0, "print2\n")
            hcl.return_(A[x] + 1)

        matrix_B = hcl.compute((m,k), lambda x, y: A[x] + B[x] + 7, "matrix_B")
        hcl.update(B, lambda x: update_B(A, x))
        matrix_C = hcl.compute((m,k), lambda x, y: A[x] + B[x] + 7, "matrix_C")

        hcl.print(0, "should not print\n")

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = np.zeros(10)
    _A = hcl.asarray(a)
    _B = hcl.asarray(b)

    # assert condition becomes false on 8th iteration
    # only print1 prints on 8th iteration
    f(_A, _B)

def test_assert_module_cond_return_if_only():

    def algorithm(A, B):

        @hcl.def_([A.shape, ()])
        def update_B(A, x):
            with hcl.if_(A[x] < 5):
                hcl.print(0, "print1\n")
                hcl.assert_(A[x] < 4, "assert message 1")
                hcl.print(0, "print2\n")
                hcl.return_(-1)
            hcl.assert_(A[x] >= 5, "assert message 2")
            hcl.print(0, "not in if\n")
            hcl.return_(A[x] + 1)
        matrix_B = hcl.compute((m,k), lambda x, y: A[x] + B[x] + 7, "matrix_B")

        hcl.update(B, lambda x: update_B(A, x))

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = np.zeros(10)
    _A = hcl.asarray(a)
    _B = hcl.asarray(b)

    # enters if statement 4 times
    # assert condition in the if statement becomes false on the 5th iteration
    f(_A, _B)

def test_assert_module_cond_return_if_else():
    hcl.init(raise_assert_exception=False)

    def algorithm(A, B):

        @hcl.def_([A.shape, ()])
        def update_B(A, x):
            with hcl.if_(A[x] > 5):
                hcl.print(0, "print if 1\n")
                hcl.assert_(A[x] <= 5, "assert in if")
                hcl.print(0, "print if 2\n")
                hcl.return_(-1)
            with hcl.else_():
                hcl.print(0, "print else 1\n")
                hcl.assert_(A[x] <= 5, "assert in else")
                hcl.print(0, "print else 2\n")
                hcl.return_(A[x] + 1)

        hcl.update(B, lambda x: update_B(A, x))
        hcl.print(0, "shouldn't be printed")
    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = np.zeros(10)
    _A = hcl.asarray(a)
    _B = hcl.asarray(b)

    # enters else statement 6 times
    # enters if statement once: assert condition in the if is false the first time entering
    f(_A, _B)

def test_assert_module_cond_return_multi_if_else():
    hcl.init(raise_assert_exception=False)

    def algorithm(A, B):

        @hcl.def_([A.shape, ()])
        def update_B(A, x):
            with hcl.if_(A[x] > 5):
                with hcl.if_(A[x] > 7):
                    hcl.print(0, "in if 1\n")
                    hcl.assert_(A[x] == 1, "assert in if")
                    hcl.print(0, "in if 2\n")
                    hcl.return_(-2)
                hcl.return_(-1)
            with hcl.else_():
                with hcl.if_(A[x] > 3):
                    hcl.print(0, "in else 1\n")
                    hcl.assert_(A[x] == 4, "assert in else")
                    hcl.print(2, "in else 2\n")
                    hcl.return_(-3)
            hcl.return_(A[x] + 1)

        hcl.update(B, lambda x: update_B(A, x))

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    b = np.zeros(10)
    _A = hcl.asarray(a)
    _B = hcl.asarray(b)

    # enters the outer else statement but not the nested if 4 times
    # enters the nested if in the else statement twice
    # assert condition for "assert in else" is false on the second iteration
    f(_A, _B)

def test_assert_module_cond_return_for():
    hcl.init(raise_assert_exception=False)

    def algorithm(A, B):

        @hcl.def_([A.shape, ()])
        def update_B(A, x):
            with hcl.for_(0, 10) as i:
                hcl.assert_(i < 20)
                hcl.print(0, "in for loop\n")
                with hcl.if_(A[x] == i):
                    hcl.assert_(A[x] > 10, "assert in if")
                    hcl.print(0, "this should not be printed")
                    hcl.return_(1)
            hcl.return_(A[x])

        hcl.update(B, lambda x: update_B(A, x))

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    a = np.array([12, 2, 3, 1, 6, 5, 2, 8, 3, 0])
    b = np.zeros(10)
    _A = hcl.asarray(a)
    _B = hcl.asarray(b)

    # enters the for loop 12 times without entering the if statement
    # enters the if statement on the 13th iteration
    # "assert in if" is false the first time entering the if statement
    f(_A, _B)

def test_assert_module_multi_calls():
    hcl.init(raise_assert_exception=False)

    def algorithm(A, B):

        @hcl.def_([A.shape, B.shape, ()])
        def add(A, B, x):
            hcl.assert_(x < 3, "assert in add")
            hcl.print(0, "in add\n")
            hcl.return_(A[x] + B[x])

        @hcl.def_([A.shape, B.shape, ()])
        def mul(A, B, x):
            temp = hcl.scalar(0)
            with hcl.for_(0, x) as i:
                hcl.assert_(x < 5, "assert in for")
                temp[0] += add(A, B, x)
                hcl.print(0, "in for\n")
            hcl.return_(temp[0])

        tmp =  hcl.compute(A.shape, lambda x: mul(A, B, x))
        hcl.print(0, "shouldn't print\n")
        return tmp

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    b = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    c = np.zeros(10)
    _A = hcl.asarray(a)
    _B = hcl.asarray(b)
    _C = hcl.asarray(c)

    # enters add function 3 times while assert condition is true
    # on fourth time entering add, the condition for assert in add is false
    f(_A, _B, _C)

def test_assert_module_declarative():
    hcl.init(raise_assert_exception=False)

    def algorithm(a, b, c):

        @hcl.def_([a.shape, b.shape, c.shape])
        def add(a, b, c):
            hcl.update(c, lambda *x: a[x] + b[x])
            hcl.assert_(False)
            hcl.print(0, "print add")

        hcl.print(0, "print1\n")
        add(a, b, c)
        hcl.print(0, "print end\n")

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

    # should only print "print1" before assert error
    # array c should still be updated
    f(_a, _b, _c)
    assert np.array_equal(_c.asnumpy(), a + b)

def test_assert_module_declarative_internal_allocate():
    hcl.init(raise_assert_exception=False)

    def algorithm(a, b, c):

        @hcl.def_([a.shape, b.shape, c.shape])
        def add(a, b, c):
            d = hcl.compute(a.shape, lambda *x: a[x] + b[x])
            hcl.assert_(False)
            hcl.print(0, "print1")
            hcl.update(c, lambda *x: d[x] + 1)
            hcl.assert_(False)
            hcl.print(0, "print2")
        tmp = hcl.compute((64, 64), lambda x, y: 4 + 8)
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

    # should immediately print assert error
    # c should not be updated
    f(_a, _b, _c)
    assert np.array_equal(_c.asnumpy(), np.zeros(10))

def test_assert_module_declarative_compute_at():
    hcl.init(raise_assert_exception=False)

    def algorithm(a, b, c):

        @hcl.def_([a.shape, b.shape, c.shape])
        def add(a, b, c):
            d = hcl.compute(a.shape, lambda *x: a[x] + b[x], "d")
            hcl.assert_(True, "assert error 1")
            hcl.print(0, "print1\n")
            hcl.update(c, lambda *x: d[x] + 1, "u")
            hcl.assert_(False, "assert error 2")
            hcl.print(0, "print2")
        tmp = hcl.compute((64, 64), lambda x, y: 4 + 8)
        add(a, b, c)
        hcl.print(0, "print end")

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

    # should only print "print1" before printing "assert error 2"
    # c should still be updated
    f(_a, _b, _c)
    assert np.array_equal(_c.asnumpy(), a + b + 1)

def test_assert_all_true():
    hcl.init(raise_assert_exception=False)

    def algorithm(a, b, c):

        @hcl.def_([a.shape, b.shape, c.shape])
        def add(a, b, c):
            with hcl.for_(0, 10) as i:
                a[i] = 0
                hcl.assert_(i < 10, "assert error 1")
            d = hcl.compute(a.shape, lambda *x: a[x] + b[x])
            hcl.assert_(a[0] == 0, "assert error 2")
            hcl.update(c, lambda *x: d[x] + 1)
            hcl.assert_(a[0] == 0, "assert error 3")

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

    # all assert conditions are true: program output should not be affected
    f(_a, _b, _c)

    assert np.array_equal(_c.asnumpy(), b + 1)

test_assert_module_no_return()
test_assert_module_with_return()
test_assert_module_cond_return_if_only()
test_assert_module_cond_return_if_else()
test_assert_module_cond_return_multi_if_else()
test_assert_module_cond_return_for()
test_assert_module_multi_calls()
test_assert_module_declarative()
test_assert_module_declarative_internal_allocate()
test_assert_module_declarative_compute_at()
test_assert_all_true()
