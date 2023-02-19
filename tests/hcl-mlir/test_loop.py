# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import os, sys
import numpy as np


def test_loop():
    A = hcl.placeholder((32, 32), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "B")
        return B

    def kernel_two(A):
        B = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "B")
        C = hcl.compute(A.shape, lambda i, j: B[i, j] + 1, "C")
        return C

    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="csyn", project="loop.prj")
    # Only when creating the schedule, kernel will be executed
    s = hcl.create_schedule([A], kernel_two)
    s_B, s_C = kernel_two.B, kernel_two.C
    # s = hcl.create_schedule([A], kernel)
    # s_B = kernel.B
    # s[s_B].reorder(s_B.axis[1], s_B.axis[0])
    # outer, inner = s[s_B].split(s_B.axis[1], factor=2)
    # x_outer, x_inner, y_outer, y_inner = s[s_B].tile(s_B.axis[0], s_B.axis[1], x_factor=2, y_factor=4)
    # fused = s[s_B].fuse(s_B.axis[0], s_B.axis[1])
    # s[s_B].pipeline(s_B.axis[0])
    # s[s_B].compute_at(s[s_C], s_C.axis[1])
    # s.partition(A, hcl.Partition.Block, dim=1, factor=2) # Block
    # s.partition(A, hcl.Partition.Cyclic, dim=2, factor=2) # Cyclic
    s.to(s_B, s[s_C])
    # s.reuse_at(A, s[s_B], s_B.axis[0])
    # s.buffer_at(A, s[s_B], s_B.axis[0])
    mod = hcl.build(s, target=target)
    print(mod.src)


if __name__ == "__main__":
    test_loop()
