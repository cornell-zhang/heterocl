# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import os, sys
import numpy as np


def test_stream():
    A = hcl.placeholder((32, 32), "A")

    def kernel_two(A):
        B = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "B")
        C = hcl.compute(A.shape, lambda i, j: B[ii, jj] + 1, "C")
        return C

    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="csyn", project="stream.prj")
    # Only when creating the schedule, kernel will be executed
    s = hcl.create_schedule([A], kernel_two)
    s_B, s_C = kernel_two.B, kernel_two.C
    s.to(s_B, s[s_C])
    mod = hcl.build(s, target=target)
    print(mod.src)
    # mod()
    report = mod.report()
    report.display()


if __name__ == "__main__":
    test_stream()
