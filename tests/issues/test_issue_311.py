import heterocl as hcl
import numpy as np

def test_for_index_casting():

    def kernel(A):
        with hcl.for_(0, 10) as i:
            with hcl.for_(i, 10) as j:
                A[j] += i

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.zeros(10)
    golden_A = np.zeros(10)

    for i in range(0, 10):
        for j in range(i, 10):
            golden_A[j] += i

    hcl_A = hcl.asarray(np_A)

    f(hcl_A)

    ret_A = hcl_A.asnumpy()
    assert np.array_equal(golden_A, ret_A)

