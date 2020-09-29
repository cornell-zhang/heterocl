import heterocl as hcl
import numpy as np

def test_slice_op():

    hcl.init()

    def kernel(A):
        return hcl.compute(A.shape, lambda x: A[x][8:0] + A[x][16:8])

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(10)
    golden = (np_A & 0xFF) + ((np_A >> 8) & 0xFF)
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)
