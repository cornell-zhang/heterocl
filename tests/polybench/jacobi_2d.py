# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import math as mt
import os


def top_jacobi_2d(N, TSTEPS, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((N, N), "A")
    B = hcl.placeholder((N, N), "B")

    def kernel_jacobi_2d(A, B):
        def update(A, B):
            with hcl.for_(1, N - 1, name="L1") as i:
                with hcl.for_(1, N - 1, name="L2") as j:
                    B[i][j] = 0.2 * (
                        A[i][j] + A[i][j - 1] + A[i][j + 1] + A[i + 1][j] + A[i - 1][j]
                    )

            with hcl.for_(1, N - 1, name="L3") as i:
                with hcl.for_(1, N - 1, name="L4") as j:
                    A[i][j] = 0.2 * (
                        B[i][j] + B[i][j - 1] + B[i][j + 1] + B[i + 1][j] + B[i - 1][j]
                    )

        hcl.mutate((TSTEPS,), lambda m: update(A, B), "main_loop")

    s = hcl.create_schedule([A, B], kernel_jacobi_2d)

    #### Apply customizations ####

    main_loop = kernel_jacobi_2d.main_loop

    s[main_loop].pipeline(main_loop.L1)
    s[main_loop].unroll(main_loop.L2)

    s[main_loop].pipeline(main_loop.L3)
    s[main_loop].unroll(main_loop.L4)

    #### Apply customizations ####

    return hcl.build(s, target=target)


import numpy as np
import math as mt
from utils.helper import *


def jacobi_2d_golden(N, TSTEPS, A, B, DATA_TYPE):
    dtype = NDATA_TYPE_DICT[DATA_TYPE.lower()]

    for t in range(TSTEPS):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                B[i][j] = 0.2 * (
                    A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] + A[i - 1][j]
                )
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                A[i][j] = 0.2 * (
                    B[i][j] + B[i][j - 1] + B[i][1 + j] + B[1 + i][j] + B[i - 1][j]
                )
