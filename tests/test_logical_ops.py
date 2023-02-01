import heterocl as hcl
import numpy as np


def test_while_with_and():
    hcl.init()

    def kernel():
        a = hcl.scalar(0, "a", dtype=hcl.UInt(8))
        b = hcl.scalar(0, "a", dtype=hcl.UInt(8))
        res = hcl.scalar(0, "res", dtype=hcl.UInt(8))
        with hcl.while_(hcl.and_(a.v == 0, b.v == 0)):
            res.v = a.v + b.v + 1
            a.v += 1

        with hcl.while_(hcl.and_(a.v == 1, b.v == 0) != 0):
            res.v += 2
            a.v += 1

        return res

    s = hcl.create_schedule([], kernel)
    print(hcl.lower(s))
    f = hcl.build(s)
    hcl_res = hcl.asarray(np.zeros((1,), dtype=np.int32), dtype=hcl.UInt(8))
    f(hcl_res)
    assert hcl_res.asnumpy()[0] == 3
