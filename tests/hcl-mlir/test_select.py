# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import os, sys
import numpy as np


def test_select():
    A = hcl.placeholder((32, 32), "A")

    def kernel(A):
        B = hcl.compute(
            A.shape,
            lambda i, j: hcl.select(hcl.and_(i < j, i == j), A[i, j], A[i, j] + 1),
            "B",
        )
        return B

    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="csim", project="select.prj")
    s = hcl.create_schedule([A], kernel)
    print(hcl.lower(s))
    # s_B = kernel.B
    # s.to(A, target.xcel)
    # s.to(s_B, target.host)
    # mod = hcl.build(s, target)
    # mod()


if __name__ == "__main__":
    test_select()
