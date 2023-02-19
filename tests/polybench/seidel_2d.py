# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import math as mt
import numpy as np


def top_seidel_2d(N, TSTEPS, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((N, N), "A")

    def kernel_seidel_2d(A):
        def update(A):
            with hcl.for_(1, N - 1, name="i") as i:
                with hcl.for_(1, N - 1, name="j") as j:
                    A[i][j] = (
                        A[i - 1][j - 1]
                        + A[i - 1][j]
                        + A[i - 1][j + 1]
                        + A[i][j - 1]
                        + A[i][j]
                        + A[i][j + 1]
                        + A[i + 1][j - 1]
                        + A[i + 1][j]
                        + A[i + 1][j + 1]
                    ) / 9.0

        hcl.mutate((TSTEPS,), lambda m: update(A), "main_loop")

    s = hcl.create_schedule([A], kernel_seidel_2d)
    #### Apply customizations ####
    main_loop = kernel_seidel_2d.main_loop
    fa = s[main_loop].fuse(main_loop.i, main_loop.j)
    s[main_loop].unroll(fa)
    #### Apply customizations ####
    return hcl.build(s, target=target)


def seidel_2d_golden(N, TSTEPS, A, DATA_TYPE):
    dtype = NDATA_TYPE_DICT[DATA_TYPE.lower()]

    for t in range(TSTEPS):
        for i in range(1, N - 2):
            for j in range(1, N - 2):
                A[i][j] = (
                    A[i - 1][j - 1]
                    + A[i - 1][j]
                    + A[i - 1][j + 1]
                    + A[i][j - 1]
                    + A[i][j]
                    + A[i][j + 1]
                    + A[i + 1][j - 1]
                    + A[i + 1][j]
                    + A[i + 1][j + 1]
                ) / (dtype)(9)
