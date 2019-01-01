import heterocl as hcl
import numpy as np

def _test_logic_op(op):

    def kernel(A, B):
        return hcl.compute(A.shape,
                lambda x: hcl.select(op(A[x]>5, B[x]>5), 0, 1))

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    return f

def test_and():

    f = _test_logic_op(hcl.and_)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.random.randint(10, size=(10,))
    np_C = np.zeros(10)

    golden_C = [0 if np_A[i]>5 and np_B[i]>5 else 1 for i in range(0, 10)]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_C = hcl.asarray(np_C)

    f(hcl_A, hcl_B, hcl_C)

    ret_C = hcl_C.asnumpy()
    assert np.array_equal(ret_C, golden_C)

def test_or():

    f = _test_logic_op(hcl.or_)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.random.randint(10, size=(10,))
    np_C = np.zeros(10)

    golden_C = [0 if np_A[i]>5 or np_B[i]>5 else 1 for i in range(0, 10)]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_C = hcl.asarray(np_C)

    f(hcl_A, hcl_B, hcl_C)

    ret_C = hcl_C.asnumpy()
    assert np.array_equal(ret_C, golden_C)

