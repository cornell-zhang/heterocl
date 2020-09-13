import heterocl as hcl
import numpy as np
import os

def test_inter_stage_stream():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda i: A[i] + 1, "B")
        C = hcl.compute(B.shape, lambda i: A[i] + B[i], "C")
        D = hcl.compute(B.shape, lambda i: C[i], "D")
        return D

    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csyn")
    s = hcl.create_schedule([A], kernel)

    s.to(A, s[kernel.B])
    s.to([A], target.xcel)
    s.to(kernel.D, target.host)

    s.to(kernel.B, s[kernel.C])
    s.to(kernel.C, s[kernel.D])

    f = hcl.build(s, target)
    np_A = np.zeros((10,))
    np_D = np.zeros((10,))
    hcl_A = hcl.asarray(np_A)
    hcl_D = hcl.asarray(np_D)
    f(hcl_A, hcl_D)

if __name__ == "__main__":
    test_inter_stage_stream()
