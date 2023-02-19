# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import math as mt
import os


def top_heat_3d(N, TSTEPS, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((N, N, N), "A")
    B = hcl.placeholder((N, N, N), "B")

    def kernel_heat_3d(A, B):
        def update(A, B, m):
            const0 = hcl.scalar(0.125)
            const1 = hcl.scalar(2.0)
            with hcl.for_(1, N - 1, name="L1") as i:
                with hcl.for_(1, N - 1, name="L2") as j:
                    with hcl.for_(1, N - 1, name="L3") as k:
                        B[i][j][k] = (
                            const0.v
                            * (A[i + 1][j][k] - const1.v * A[i][j][k] + A[i - 1][j][k])
                            + const0.v
                            * (A[i][j + 1][k] - const1.v * A[i][j][k] + A[i][j - 1][k])
                            + const0.v
                            * (A[i][j][k + 1] - const1.v * A[i][j][k] + A[i][j][k - 1])
                            + A[i][j][k]
                        )

            with hcl.for_(1, N - 1, name="L4") as i:
                with hcl.for_(1, N - 1, name="L5") as j:
                    with hcl.for_(1, N - 1, name="L6") as k:
                        A[i][j][k] = (
                            const0.v
                            * (B[i + 1][j][k] - const1.v * B[i][j][k] + B[i - 1][j][k])
                            + const0.v
                            * (B[i][j + 1][k] - const1.v * B[i][j][k] + B[i][j - 1][k])
                            + const0.v
                            * (B[i][j][k + 1] - const1.v * B[i][j][k] + B[i][j][k - 1])
                            + B[i][j][k]
                        )

        hcl.mutate((TSTEPS,), lambda m: update(A, B, m + 1), "main_loop")

    s = hcl.create_schedule([A, B], kernel_heat_3d)

    #### Apply customizations ####

    main_loop = kernel_heat_3d.main_loop

    s[main_loop].pipeline(main_loop.L1)
    s[main_loop].unroll(main_loop.L2)

    s[main_loop].pipeline(main_loop.L4)
    s[main_loop].unroll(main_loop.L5)

    #### Apply customizations ####

    return hcl.build(s, target=target)


import numpy as np
import math as mt


def heat_3d_golden(N, TSTEPS, A, B, DATA_TYPE):
    dtype = NDATA_TYPE_DICT[DATA_TYPE.lower()]

    for t in range(TSTEPS):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    B[i][j][k] = (
                        0.125 * (A[i + 1][j][k] - 2.0 * A[i][j][k] + A[i - 1][j][k])
                        + 0.125 * (A[i][j + 1][k] - 2.0 * A[i][j][k] + A[i][j - 1][k])
                        + 0.125 * (A[i][j][k + 1] - 2.0 * A[i][j][k] + A[i][j][k - 1])
                        + A[i][j][k]
                    )

        for i in range(1, N - 1):
            for j in range(1, N - 1):
                for k in range(1, N - 1):
                    A[i][j][k] = (
                        0.125 * (B[i + 1][j][k] - 2.0 * B[i][j][k] + B[i - 1][j][k])
                        + 0.125 * (B[i][j + 1][k] - 2.0 * B[i][j][k] + B[i][j - 1][k])
                        + 0.125 * (B[i][j][k + 1] - 2.0 * B[i][j][k] + B[i][j][k - 1])
                        + B[i][j][k]
                    )
