import heterocl as hcl
import numpy as np
from itertools import permutations
import os
import sys

def test_kernel_in_kernel():
    hcl.init()
    dtype=hcl.Int()
    A = hcl.placeholder((10,), dtype=dtype, name="A")
    B = hcl.placeholder((10,), dtype=dtype, name="B")

    def kernel(A, B):
        @hcl.def_([()])
        def popcount(value):
            count = hcl.scalar(0, "count")
            numb = hcl.scalar(value, "numb", dtype=hcl.UInt(32))
            with hcl.for_(0, 32, name="i") as i:
                count.v += numb.v & 1
                numb.v >>= 1
            hcl.return_(count.v)

        C = hcl.compute((10,), lambda x: A[x] + B[x], "C")
        hcl.update(C, lambda x: popcount(C[x]), "updateC")

    p = hcl.platform.aws_f1
    p.config(compile="vitis", mode="debug")
    s = hcl.create_schedule([A, B], kernel)   

    s.to([A, B], p.xcel)
    s.to(kernel.updateC.C, p.host)
    # print(kernel.popcount._op.op.body)
    print(hcl.build(s, p))    

def test_tensor_layout():
    m=64
    n=64
    k=64
    hcl.init()
    dtype=hcl.Int()

    A = hcl.placeholder((m,k), dtype=dtype, name="A")
    B = hcl.placeholder((k,n), dtype=dtype, name="B")

    def kernel(A, B):
        Y = hcl.compute((m, n), lambda *args: 0, name="Y0")
        with hcl.Stage("Y"):
            with hcl.for_(0, m, name="i") as i:
                with hcl.for_(0, n, name="j") as j:
                    Y[i][j] = 0
                    with hcl.for_(0, k, name="k") as r:
                        Y[i][j] += A[i][r] * B[r][j]

    p = hcl.platform.aws_f1
    p.config(compile="vitis", mode="debug")
    s = hcl.create_schedule([A, B], kernel)

    # Output tensor Y0 is initialized on host
    s.to([A, B, kernel.Y0], p.xcel)
    s.to(kernel.Y.Y0, p.host)

    # Default systolic array
    s[kernel.Y].systolic()

    # Tranpose the tensor B before stage Y
    s.transpose(kernel.Y.B)

    # Pack the input tensors
    s.pack(kernel.Y.B, factor=512)
    s.pack(kernel.Y.A, factor=512)
    print(hcl.lower(s))
    print(hcl.build(s, p))
    
    if os.system("which v++ >> /dev/null") != 0:
        return 
        
    p.config(compile="vitis", mode="sw_sim")
    np_A = np.random.randint(0, 10, A.shape)
    np_B = np.random.randint(0, 10, B.shape)
    np_O = np.zeros((m,n))

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_O = hcl.asarray(np_O)
    f = hcl.build(s, p)
    f(hcl_A, hcl_B)


if __name__ == "__main__":
    # test_kernel_in_kernel()
    test_tensor_layout()
