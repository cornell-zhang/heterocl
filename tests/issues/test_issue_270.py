import numpy as np
import heterocl as hcl
import time
import os
import random

def test_duplicated():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        B = hcl.compute(A.shape, 
                lambda i: A[i] + 1, "B")
        C = hcl.compute(B.shape,
                lambda i: B[i] + 1, "C")
        return C

    target = hcl.Platform.zc706
    target.config(compiler="vivado_hls", mode="csyn")
    s = hcl.create_schedule([A], kernel)
    s.to([A], target.xcel)
    s.to(kernel.C, target.host)

    s.to(kernel.B, s[kernel.C])
    # ignored duplicated streaming
    s.to(kernel.B, s[kernel.C]) 

    f = hcl.build(s, target)
    np_A = np.zeros((10,))
    np_C = np.zeros((10,))
    hcl_A = hcl.asarray(np_A)
    hcl_C = hcl.asarray(np_C)
    f(hcl_A, hcl_C)

if __name__ == '__main__':
    test_duplicated()
