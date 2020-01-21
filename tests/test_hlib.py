import heterocl as hcl
import numpy as np
import hlib

def test_sort():

    hcl.init()
    a_np = np.random.randint(low=0, high=20, size=(10,3)) 
    b_np = np.zeros((10,3)) 

    A = hcl.placeholder((10, 3), name="A")
    B = hlib.math.sort(A, axis=1, name="B")
    s = hcl.create_schedule([A, B])

    print(hcl.lower(s))
    f = hcl.build(s)

    a_hcl = hcl.asarray(a_np)
    b_hcl = hcl.asarray(b_np)

    f(a_hcl, b_hcl)

    ret_b = b_hcl.asnumpy()
    sorted_b = np.sort(a_np, axis=1)

    assert np.array_equal(ret_b, sorted_b)


def test_argmax():

    hcl.init()
    a_np = np.random.randint(low=0, high=20, size=(10,3)) 
    b_np = np.zeros((10,2)) 

    A = hcl.placeholder((10,3), name="A")
    B = hlib.math.argmax(A, axis=1, name="B")
    s = hcl.create_schedule([A, B])

    f = hcl.build(s)

    a_hcl = hcl.asarray(a_np)
    b_hcl = hcl.asarray(b_np)

    f(a_hcl, b_hcl)

    ret_b = b_hcl.asnumpy()
    argmax_b = np.argmax(a_np, axis=1)
    assert np.array_equal(ret_b[:,0], argmax_b)

# reuse hlib sort 
def test_sort_module():
    def kernel(A, B):
        def freduce(x, Y): # passed-in .v.s reducer
            with hcl.for_(0, 3) as i:
                with hcl.if_(x < Y[i]):
                    with hcl.for_(2, i, -1) as j:
                        Y[j] = Y[j-1]
                    Y[i] = x
                    hcl.break_()
    
        @hcl.def_([(10,3), (10,3)])
        def sreduce(A, B):
            init = hcl.compute((3,), lambda x: 11)
            my_sort = hcl.reducer(init, freduce)
            r = hcl.reduce_axis(0, 3, name="rdx")
            hcl.update(B, lambda x, _y: my_sort(A[x, r], axis=r))
    
        sreduce(A, B)
    
    A = hcl.placeholder((10, 3))
    B = hcl.placeholder((10, 3))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)
    
    np_A = np.random.randint(10, size=(10,3))
    np_B = np.random.randint(10, size=(10,3))
    
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    f(hcl_A, hcl_B)

    ret_b = hcl_B.asnumpy()
    sorted_b = np.sort(np_A, axis=1)
    assert np.array_equal(ret_b, sorted_b)

# reuse hlib sort 
def test_sort_function():

    def kernel(A, B):
        hlib.function.sort(A, B)

    hcl.init()
    a_np = np.random.randint(low=0, high=20, size=(10,3)) 
    b_np = np.zeros((10,3)) 

    A = hcl.placeholder((10, 3), name="A")
    B = hcl.placeholder((10, 3), name="B")
    s = hcl.create_schedule([A, B], kernel)

    f = hcl.build(s)

    a_hcl = hcl.asarray(a_np)
    b_hcl = hcl.asarray(b_np)

    f(a_hcl, b_hcl)

    ret_b = b_hcl.asnumpy()
    sorted_b = np.sort(a_np, axis=1)
    assert np.array_equal(ret_b, sorted_b)

# reuse hlib argmax 
def test_argmax_function():

    def kernel(A, B):
        hlib.function.argmax(A, B)

    hcl.init()
    a_np = np.random.randint(low=0, high=20, size=(10,3)) 
    b_np = np.zeros((10,2)) 

    A = hcl.placeholder((10, 3), name="A")
    B = hcl.placeholder((10, 2), name="B")
    s = hcl.create_schedule([A, B], kernel)

    f = hcl.build(s)

    a_hcl = hcl.asarray(a_np)
    b_hcl = hcl.asarray(b_np)

    f(a_hcl, b_hcl)

    ret_b = b_hcl.asnumpy()
    argmax_b = np.argmax(a_np, axis=1)
    assert np.array_equal(ret_b[:,0], argmax_b)

