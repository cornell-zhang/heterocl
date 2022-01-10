import heterocl as hcl
import os, sys
import numpy as np


def test_loop():

    A = hcl.placeholder((32, 32), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda i, j : A[i, j] + 1.0, "B")
        return B

    target = None # hcl.platform.zc706
    # Only when creating the schedule, kernel will be executed
    s = hcl.create_schedule([A], kernel)
    s_B = kernel.B
    s[s_B].reorder(s_B.axis[1], s_B.axis[0])
    s[s_B].pipeline(s_B.axis[0])
    f = hcl.build(s, target)
    print(f)

if __name__ == "__main__":
    test_loop()