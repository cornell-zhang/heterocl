import heterocl as hcl
import numpy as np
from itertools import permutations
import os, sys
import argparse

def autosa_systolic_array(size):
    m = size
    n = size
    k = size

    dtype=hcl.Float()
    hcl.init(dtype)

    A = hcl.placeholder((m,k), dtype=dtype, name="A")
    B = hcl.placeholder((k,n), dtype=dtype, name="B")

    def kernel(A, B):
        Y = hcl.compute((m, n), lambda *args: 0, dtype=dtype, name="Y0")
        with hcl.Stage("Y"):
            with hcl.for_(0, m, name="i") as i:
                with hcl.for_(0, n, name="j") as j:
                    Y[i][j] = 0
                    with hcl.for_(0, k, name="k") as r:
                        Y[i][j] += A[i][r] * B[r][j]
        return Y

    # Note that you have to make sure AutoSA binary
    # in on the PATH by running which command, otherwise HCL runtime
    # will only generate a function placeholder for the GEMM kernel
    p = hcl.Platform.xilinx_zc706
    p.config(compiler="vitis", mode="csyn")
    
    s = hcl.create_schedule([A, B], kernel)
    MM = kernel.Y

    s.to([A, B, kernel.Y0], p.xcel)
    s.to(kernel.Y.Y0, p.host)

    s[kernel.Y].systolic()
    s.transpose(kernel.Y.B)
    s.pack([MM.B, MM.A, MM.Y0], factor=512)

    np_A = np.random.randint(10, size=(m,k))
    np_B = np.random.randint(10, size=(k,n))
    np_C = np.zeros((m,n))
    args = (np_A, np_B, np_C)

    print(hcl.lower(s))
    f = hcl.build(s, target=p)
    f(hcl.asarray(np_A), hcl.asarray(np_B), hcl.asarray(np_C))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', nargs='?', const=1024, type=int, default=1024)
    args = parser.parse_args()
    autosa_systolic_array(args.size)
