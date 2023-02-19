# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import os, sys
import numpy as np
import pytest


def test_systolic():
    A = hcl.placeholder((32, 32), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "B")
        # C = hcl.compute(A.shape, lambda i, j : B[i, j] + 1, "C")
        k = hcl.reduce_axis(0, 32, "k")
        C = hcl.compute(A.shape, lambda i, j: hcl.sum(A[i, k] * B[k, j], axis=k), "C")
        return C

    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="csim", project="systolic.prj")
    s = hcl.create_schedule([A], kernel)
    s_B, s_C = kernel.B, kernel.C
    s[s_C].systolic()
    s.to(A, target.xcel)
    # s.to(s_B, s[s_C])
    s.to(s_C, target.host)
    mod = hcl.build(s, target)
    # mod()


if __name__ == "__main__":
    test_systolic()
