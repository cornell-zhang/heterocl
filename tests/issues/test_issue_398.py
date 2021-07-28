import heterocl as hcl
import numpy as np

def test_mask():

    def mask(A):
        return hcl.compute(A.shape, lambda x: (A[x] & 0xFFFF), "mask", dtype=A.dtype)

    A = hcl.placeholder((2,), "A", dtype=hcl.UInt(16))
    s = hcl.create_schedule([A], mask)

    m = hcl.build(s)

    hcl_A = hcl.asarray([10,5], dtype=A.dtype)
    hcl_R = hcl.asarray([99,99], dtype=hcl.UInt(16))
    m(hcl_A, hcl_R)
    assert np.array_equal(hcl_A.asnumpy(), [10, 5])
