# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np


def test_host_xcel():
    A = hcl.placeholder((10, 32), "A")

    def kernel(A):
        C = hcl.compute((10, 32), lambda i, j: A[i, j] + 1, "C")
        D = hcl.compute((10, 32), lambda i, j: C[i, j] * 2, "D")
        E = hcl.compute((10, 32), lambda i, j: D[i, j] * 3, "E")
        return E

    target = hcl.Platform.aws_f1
    s = hcl.create_schedule([A], kernel)

    s.to([A], target.xcel)
    s.to([kernel.E], target.host)
    s.to(kernel.C, s[kernel.D], fifo_depth=1)
    s.to(kernel.D, s[kernel.E], fifo_depth=1)
    # s.to(kernel.C, target.xcel)
    # s.to(kernel.D, target.host)

    target.config(compiler="vivado_hls", mode="csyn", project="host-xcel.prj")
    mod = hcl.build(s, target)
    mod()


if __name__ == "__main__":
    test_host_xcel()
