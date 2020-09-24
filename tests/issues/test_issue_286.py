import heterocl as hcl
import numpy as np
import os

def test_tile():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    shape = (10,32)
    A = hcl.placeholder(shape, "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda *i: A[i] + 1, "B")
        C = hcl.compute(A.shape, lambda *i: B[i] * 2, "C")
        return C

    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csyn")
    s = hcl.create_schedule([A], kernel)

    s.to([A], target.xcel)
    s.to(kernel.C, target.host)

    stage = kernel.C
    yo, yi = s[stage].split(stage.axis[0], factor=3)
    xo, xi = s[stage].split(stage.axis[1], factor=3)

    # create streaming arrays within tiles
    s.to(kernel.B, s[kernel.C])

    print(hcl.lower(s))

if __name__ == '__main__':
    test_tile()
