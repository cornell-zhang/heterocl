import heterocl as hcl
import numpy as np

def test_issue_410():

    A = hcl.Struct ({'foo': hcl.UInt(16) })
    B = hcl.Struct ({'foo': 'uint16' })

    assert A['foo'] == B['foo']

def test_issue_410_2():

    A = hcl.placeholder((10,), dtype="uint16")

    def f(A):
        t = hcl.Struct ({'foo': 'uint16' })
        B = hcl.compute(A.shape, lambda x: (A[x],), dtype=t)
        return hcl.compute(A.shape, lambda x: B[x].foo)

    s = hcl.create_schedule([A], f)
    f = hcl.build(s)

    npA = np.random.randint(0, 10, A.shape)
    npO = np.zeros(A.shape)

    hclA = hcl.asarray(npA, dtype="uint16")
    hclO = hcl.asarray(npO)

    f(hclA, hclO)

    assert np.array_equal(npA, hclO.asnumpy())
