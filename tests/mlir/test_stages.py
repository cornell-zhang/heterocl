import heterocl as hcl
import os, sys
import numpy as np


def test_stages():

    A = hcl.placeholder((32, 32), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda i, j : A[i, j] + 1, "B")
        C = hcl.compute(A.shape, lambda i, j : A[i, j] + 1, "C")
        D = hcl.compute(A.shape, lambda i, j : B[i, j] + 1, "D")
        E = hcl.compute(A.shape, lambda i, j : C[i, j] + 1, "E")
        F = hcl.compute(A.shape, lambda i, j : D[i, j] + E[i, j], "F")
        return F

    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="csyn", project="stages.prj")
    s = hcl.create_schedule([A], kernel)
    mod = hcl.build(s, target=target)
    print(mod.src)
    s._DataflowGraph.dump()
    s._DataflowGraph.visualize()

if __name__ == "__main__":
    test_stages()