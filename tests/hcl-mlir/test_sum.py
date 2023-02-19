# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import os, sys
import numpy as np

dtype = hcl.UInt(12)


def test_loop():
    hcl.init(hcl.UInt(12))
    A = hcl.placeholder((32, 32), "A")
    B = hcl.placeholder((32, 32), "B")

    def gemm(A, B):
        k = hcl.reduce_axis(0, 32, "k")
        C = hcl.compute(
            (32, 32),
            lambda i, j: hcl.sum(hcl.cast(dtype, A[i, k]), axis=k, dtype=dtype),
            "C",
            dtype=dtype,
        )
        # hcl.sum(A[i / 2, k % 2] * B[k / 3, j % 3], axis=k) + 10, "C")
        D = hcl.compute(
            (32, 32),
            lambda i, j: hcl.sum(hcl.cast(dtype, C[i, k]), axis=k, dtype=dtype),
            "D",
            dtype=dtype,
        )
        return D

    # Only when creating the schedule, kernel will be executed
    s = hcl.create_schedule([A, B], gemm)
    print(s.device_module)
    mod = hcl.build(s, target="vhls")
    print(mod)


if __name__ == "__main__":
    test_loop()
