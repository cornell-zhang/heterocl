import heterocl as hcl
import os, sys
import numpy as np

def test_gemm():

    A = hcl.placeholder((32, 32), "A")
    B = hcl.placeholder((32, 32), "B")

    def gemm(A, B):
        k = hcl.reduce_axis(0, 32, "k")
        C = hcl.compute((32, 32), lambda i, j:
                hcl.sum(A[i, k] * B[k, j], axis=k), "C")
        return C

    target = None # hcl.platform.zc706
    s = hcl.create_schedule([A, B], gemm)
    f = hcl.build(s, target)
    print(f)

if __name__ == "__main__":
    test_gemm()