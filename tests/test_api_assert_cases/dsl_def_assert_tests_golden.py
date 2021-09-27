import numpy as np

def test_no_return(a, b):
    def test_update_B(A, B, x):
        print("print1")
        assert x < 2, "assert error 1"
        print("print2")
        assert x < 1, "assert error 2"
        print("print3")
        B[x] = A[x] + 1

    for i in range(0, 10):
        for z in range(0, 10):
            test_update_B(a, b, i)
    print("end")

a = np.random.randint(100, size=(10,))
b = np.zeros(10)

try:
    test_no_return(a, b)
except AssertionError as error:
    print(error)

def test_with_return(a, b):
    def update_B_test(A, x):
        print("print1")
        assert A[x] != 7, "assert error"
        print("print2")
        return A[x] + 1

    for x in range(0, 10):
        b[x] = update_B_test(a, x)

    print("should not print")

a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.zeros(10)

try:
    test_with_return(a, b)
except AssertionError as error:
    print(error)

def test_cond_return_if_only(a, b):
    def update_B_test(A, x):
        if A[x] < 5:
            print("print1")
            assert A[x] < 4, "assert message 1"
            print("print2")
            return -1
        assert A[x] >= 5, "assert message 2"
        print("not in if")
        return A[x] + 1

    for x in range (0,10):
        b[x] = update_B_test(a,x)

a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.zeros(10)

try:
    test_cond_return_if_only(a, b)
except AssertionError as error:
    print(error)

def test_assert_module_cond_return_if_else(a, b):
    def update_B_test(A, x):
        if A[x] > 5:
            print("print if 1")
            assert A[x] <= 5, "assert in if"
            print("print if 2")
            return -1
        else:
            print("print else 1")
            assert A[x] <= 5, "assert in else"
            print("print else 2")
            return A[x] + 1
    for x in range(0, 10):
        b[x] = update_B_test(a, x)
    print("shouldn't be printed")

a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.zeros(10)
try:
    test_assert_module_cond_return_if_else(a, b)
except AssertionError as error:
    print(error)
    
def test_assert_module_cond_return_multi_if_else(a, b):
    def update_B_test(A, x):
        if A[x] > 5:
            if A[x] > 7:
                print("in if 1")
                assert A[x] == 1, "assert in if"
                print("in if 2")
                return -2
            return -1
        else:
            if A[x] > 3:
                print("in else 1")
                assert A[x] == 4, "assert in else"
                print("in else 2")
                return -3
        return A[x] + 1
    for x in range(0, 10):
        b[x] = update_B_test(a, x)

a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.zeros(10)
try:
    test_assert_module_cond_return_multi_if_else(a, b)
except AssertionError as error:
    print(error)

def test_assert_module_cond_return_for(a, b):
    def update_B_test(A, x):
        for i in range(0, 10):
            assert i < 20, "assert error"
            print("in for loop")
            if A[x] == i:
                assert A[x] > 10, "assert in if"
                print("this should not be printed")
                return 1
        return A[x]
    for x in range(0, 10):
        b[x] = update_B_test(a, x)

a = np.array([12, 2, 3, 1, 6, 5, 2, 8, 3, 0])
b = np.zeros(10)

try:
    test_assert_module_cond_return_for(a, b)
except AssertionError as error:
    print(error)

def test_assert_module_multi_calls(a, b, c):
    def add_test(A, B, x):
        assert x < 3, "assert in add"
        print("in add")
        return A[x] + B[x]
    def mul_test(A, B, x):
        temp = 0
        for i in range(0, x):
            assert x < 5, "assert in for"
            temp += add_test(A, B, x)
            print("in for")
        return temp

    tmp = np.zeros(10)
    for x in range(0, 10):
        tmp = mul_test(a, b, x)
    print("shouldn't print")
    return tmp

a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
b = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
c = np.zeros(10)
try:
    test_assert_module_multi_calls(a, b, c)
except AssertionError as error:
    print(error)

def test_assert_module_declarative(a, b, c):
    def add_test(a, b, c):
        for x in range(0, 10):
            c[x] = a[x] + b[x]
        assert False, "assert error"
        print("print add")
    print("print1")
    add_test(a, b, c)
    print("print end")

a = np.random.randint(100, size=(10,))
b = np.random.randint(100, size=(10,))
c = np.zeros(10)

try:
    test_assert_module_declarative(a, b, c)
except AssertionError as error:
    print(error)

def test_assert_module_declarative_internal_allocate(a, b, c):
    def add_test(a, b, c):
        d = np.zeros(10)
        for x in range(0, 10):
            d[x] = a[x] + b[x]
        assert False, "assert error"
        print("print1")
        for x in range(0, 10):
            c[x] = d[x] + 1
        assert False, "assert error"
        print("print2")
    add_test(a, b, c)

a = np.random.randint(100, size=(10,))
b = np.random.randint(100, size=(10,))
c = np.zeros(10)

try:
    test_assert_module_declarative_internal_allocate(a, b, c)
except AssertionError as error:
    print(error)

def test_assert_module_declarative_compute_at(a, b, c):
    def add_test(a, b, c):
        d = np.zeros(10)
        for x in range(0, 10):
            d[x] = a[x] + b[x]
        assert True, "assert error 1"
        print("print1")
        for x in range(0, 10):
            c[x] = d[x] + 1
        assert False, "assert error 2"
        print("print2")
    add_test(a, b, c)
    print("print end")

a = np.random.randint(100, size=(10,))
b = np.random.randint(100, size=(10,))
c = np.zeros(10)
try:
    test_assert_module_declarative_compute_at(a, b, c)
except AssertionError as error:
    print(error)
