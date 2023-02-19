# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import os
import numpy as np


def top_trmm(M=20, N=30, alpha=1.5, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((M, M), "A")
    B = hcl.placeholder((M, N), "B")

    def kernel_trmm(A, B):
        with hcl.Stage("loop_1"):
            with hcl.for_(0, M, name="i") as i:
                with hcl.for_(0, N, name="j") as j:
                    with hcl.for_(i + 1, M, name="k") as k:
                        B[i][j] += A[k][i] * B[k][j]
                    B[i][j] = alpha * B[i][j]

    s = hcl.create_schedule([A, B], kernel_trmm)

    #### Apply customizations ####

    #### Apply customizations ####

    return hcl.build(s, target=target)


def trmm_golden(alpha, M, N, A, B, DATA_TYPE):
    dtype = NDATA_TYPE_DICT[DATA_TYPE.lower()]

    for i in range(M):
        for j in range(N):
            for k in range(i + 1, M):
                B[i][j] += A[k][i] * B[k][j]
            B[i][j] = alpha * B[i][j]
