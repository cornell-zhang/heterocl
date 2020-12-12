import heterocl as hcl
import numpy as np
import os

def test_condition_pipe():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    A = hcl.placeholder((10,), "A")

    def kernel(A):
        B = hcl.compute(A.shape, 
                lambda i: A[i] + 1, "B")
        C = hcl.compute((11,),
                lambda i: hcl.select(i < 10, B[i], 0),"C")
        return C

    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csyn")
    s = hcl.create_schedule([A], kernel)

    s.to([A], target.xcel)
    s.to(kernel.C, target.host)
    s.to(kernel.B, s[kernel.C])

    print(hcl.lower(s))


def test_zero():
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda i: A[i] + 1, "B")
        C1 = hcl.compute(A.shape, lambda i: 0, "C1")
        C2 = hcl.compute(A.shape, lambda i: 0, "C2")
        def foo(i):
            C1[i] = B[i] + 1
            C2[i] = C1[i] + 1
        hcl.mutate((10,), lambda i: foo(i), "C")
        D = hcl.compute(A.shape, lambda i: C2[i] + 1, "D")
        return D

    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csim")
    s = hcl.create_schedule([A], kernel)

    s.to([A], target.xcel)
    s.to(kernel.D, target.host)

    s.to(kernel.B, s[kernel.C])
    s.to(kernel.C.C2, s[kernel.D])

    print(hcl.lower(s))

if __name__ == '__main__':
    test_condition_pipe()
    test_zero()
