import heterocl as hcl
import numpy as np

def test_zero_allocate():

    def kernel(A):
        with hcl.for_(0, 10) as i:
            with hcl.for_(i, 10) as j:
                A[j] += i
        return hcl.compute((0,), lambda x: A[x], "B")

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    p = hcl.Platform.aws_f1
    p.config(compiler="vitis", mode="debug", backend="vhls")
    try:
        f = hcl.build(s, p)
    except:
        print("passed")

