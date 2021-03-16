import heterocl as hcl
import numpy as np
from itertools import permutations
import os, sys
import argparse

def autosa_cnn(size):
    O = 16
    I = 16
    R = 8
    C = 8
    K = 3

    hcl.init()
    dtype=hcl.Int()

    # Input/output fmaps in NHWC layout
    A = hcl.placeholder((R+K-1,C+K-1,I), dtype=dtype, name="A")
    W = hcl.placeholder((O,K,K,I), dtype=dtype, name="W")

    def kernel(A, W):
        Y = hcl.compute((R,C,O), lambda *args: 0, name="Y0")
        with hcl.Stage("Y"):
            with hcl.for_(0, O, name="o") as o:
                with hcl.for_(0, R, name="r") as r:
                    with hcl.for_(0, C, name="c") as c:
                        Y[r,c,o] = 0
                        with hcl.for_(0, I, name="i") as i:
                            with hcl.for_(0, K, name="p") as p:
                                with hcl.for_(0, K, name="q") as q:
                                    Y[r][c][o] = Y[r][c][o] + A[r+p,c+q,i] * W[o][p][q][i]
        return Y

    p = hcl.Platform.u280
    p.config(compile="vitis", mode="hw_sim")

    s = hcl.create_schedule([A, W], kernel)
    CNN = kernel.Y

    # Output tensor Y0 is initialized on host
    s.to([A, W, kernel.Y0], p.xcel)
    s.to(kernel.Y.Y0, p.host)
    s.pack([CNN.W, CNN.A, CNN.Y0], factor=512)

    # Generate SA using AutoSA
    s[kernel.Y].systolic(
        SA_ARRAY_PAR="[8,8,4,8]",
        SA_LAT_HIDING="[4,2,4]",
        SA_SPACE_TIME="[4]",
        SA_SIMD="[1,1,1,2]",
        SA_SIMD_INFO="cnn"
    )

    # Generate inputs
    np_A = np.random.randint(10, size=A.shape)
    np_W = np.random.randint(10, size=W.shape)
    np_C = np.zeros((R,C,O))
    args = (np_A, np_W, np_C)

    # Print lowered IR and inspect code
    print(hcl.lower(s))
    f = hcl.build(s, target=p)
    f.inspect(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', nargs='?', const=1024, type=int, default=1024)
    args = parser.parse_args()
    autosa_cnn(args.size)
