# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import os, sys
import numpy as np
import pytest

hcl.init(hcl.Int(12))


def test_bit():
    A = hcl.placeholder((32, 12), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda i, j: A[i, j][j], "B", hcl.Int(1))
        return B

    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="csim", project="bit.prj")
    s = hcl.create_schedule([A], kernel)
    s_B = kernel.B
    s.to(A, target.xcel)
    s.to(s_B, target.host)
    mod = hcl.build(s, target)
    mod()


if __name__ == "__main__":
    test_bit()
