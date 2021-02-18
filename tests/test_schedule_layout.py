import heterocl as hcl
import numpy as np
from itertools import permutations
import os
import sys

def test_tensor_layout():
    m=64
    n=64
    k=64
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

    # Tranpose the tensor B before stage Y
    s.transpose(kernel.Y.B)

    # Pack the input tensors
    s.pack(kernel.Y.B, factor=512)
    s.pack(kernel.Y.A, factor=512)

    print(hcl.lower(s))

if __name__ == "__main__":
    test_tensor_layout()