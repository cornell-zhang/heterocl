import heterocl as hcl
import os
import sys
import numpy as np


def test_stages():

    A = hcl.placeholder((32, 32), "A")
    # C = hcl.placeholder((32, 32), "C")

    def kernel(A):
        B = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "B")
        C = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "C")
        D = hcl.compute(A.shape, lambda i, j: B[i, j] + 1, "D")
        E = hcl.compute(A.shape, lambda i, j: C[i, j] + 1, "E")
        F = hcl.compute(A.shape, lambda i, j: D[i, j] + E[i, j], "F")
        return F

    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="debug",
                  project="stages-depth-1-new.prj")
    s = hcl.create_schedule([A], kernel)
    s.to(A, target.xcel)
    s.to(kernel.B, s[kernel.D], fifo_depth=1)
    s.to(kernel.C, s[kernel.E], fifo_depth=1)
    s.to(kernel.D, s[kernel.F], fifo_depth=1)
    s.to(kernel.E, s[kernel.F], fifo_depth=1)
    s.to(kernel.F, target.host)
    mod = hcl.build(s, target=target)
    print(mod.src)
    mod()
    # np_A = np.zeros((32, 32))
    # np_C = np.zeros((32, 32))
    # np_F = np.zeros((32, 32))
    # hcl_A = hcl.asarray(np_A)
    # hcl_C = hcl.asarray(np_C)
    # hcl_F = hcl.asarray(np_F)
    # mod(hcl_A, hcl_C)
    # report = mod.report()
    # report.display()


def test_outline():

    A = hcl.placeholder((32, 32), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "B")
        C = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "C")
        D = hcl.compute(A.shape, lambda i, j: B[i, j] + C[i, j], "D")
        return D

    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="debug",
                  project="stages-outline.prj")
    s = hcl.create_schedule([A], kernel)
    s[kernel.B].pipeline(kernel.B.axis[1])
    s.partition(kernel.B, dim=2)
    func_B = s[kernel.B].outline()
    func_C = s[kernel.C].outline()
    func_D = s[kernel.D].outline()
    print(hcl.lower(s))

    mod = hcl.build(s, top=[func_B, func_C, func_D], target=target)
    mod()


if __name__ == "__main__":
    # test_stages()
    test_outline()
