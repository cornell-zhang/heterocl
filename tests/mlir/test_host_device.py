import heterocl as hcl
import os, sys
import numpy as np


def test_host_device():

    A = hcl.placeholder((10, 32), "A")

    def kernel(A):
        C = hcl.compute((10, 32), lambda i, j: A[i, j] + 1, "C")
        D = hcl.compute((10, 32), lambda i, j: C[i, j] * 2, "D")
        E = hcl.compute((10, 32), lambda i, j: D[i, j] * 3, "E")
        return E

    target = hcl.Platform.aws_f1
    s = hcl.create_schedule([A], kernel)

    s.to([A], target.xcel)
    s.to(kernel.D, target.host)
    # s.to(kernel.C, s[kernel.D])

    target.config(compiler="vivado_hls", mode="csyn", project="host-device.prj")
    mod = hcl.build(s, target)
    mod()

if __name__ == "__main__":
    test_host_device()