# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import math as mt
import os


def top_jacobi_1d(N, TSTEPS, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((N,), "A")
    B = hcl.placeholder((N,), "B")

    def kernel_jacobi_1d(A, B):
        def update(A, B):
            with hcl.for_(1, N - 1, name="L1") as i:
                B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1])

            with hcl.for_(1, N - 1, name="L2") as i:
                A[i] = 0.33333 * (B[i - 1] + B[i] + B[i + 1])

        hcl.mutate((TSTEPS,), lambda m: update(A, B), "main_loop")

    s = hcl.create_schedule([A, B], kernel_jacobi_1d)

    #### Apply customizations ####

    main_loop = kernel_jacobi_1d.main_loop

    s[main_loop].unroll(main_loop.L1)
    s[main_loop].unroll(main_loop.L2)

    #### Apply customizations ####
    return hcl.build(s, target=target)


import numpy as np


def jacobi_1d_golden(N, TSTEPS, A, B, DATA_TYPE):
    for t in range(TSTEPS):
        for i in range(1, N - 1):
            B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1])
        for i in range(1, N - 1):
            A[i] = 0.33333 * (B[i - 1] + B[i] + B[i + 1])
