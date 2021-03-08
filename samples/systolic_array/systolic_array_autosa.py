import heterocl as hcl
import numpy as np
from itertools import permutations
import os
import sys

def autosa_systolic_array():
    m=1024
    n=1024
    k=1024
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
        return Y

    p = hcl.Platform.u280

    # [Important] Note that you have to make sure `autosa` binary
    # in on the PATH, otherwise HCL runtime will only generate a 
    # function placeholder for the GEMM code
    p.config(compile="vitis", mode="hw_sim")
    s = hcl.create_schedule([A, B], kernel)
    MM = kernel.Y

    # Output tensor Y0 is initialized on host
    s.to([A, B, kernel.Y0], p.xcel)
    s.to(kernel.Y.Y0, p.host)

    # Generate SA using AutoSA
    s[kernel.Y].systolic()

    # Tranpose the tensor B before stage Y
    s.transpose(kernel.Y.B)

    # Pack the input tensors
    s.pack([MM.B, MM.A, MM.Y0], factor=512)

    # Generate inputs
    np_A = np.random.randint(10, size=(m,k))
    np_B = np.random.randint(10, size=(k,n))
    np_C = np.zeros((m,n))
    args = (np_A, np_B, np_C)

    # Print lowered IR and inspect code
    print(hcl.lower(s))
    f = hcl.build(s, target=p)
    f.inspect(args)

if __name__ == '__main__':
    autosa_systolic_array()
