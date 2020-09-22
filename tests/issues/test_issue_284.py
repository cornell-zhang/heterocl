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

if __name__ == '__main__':
    test_condition_pipe()
