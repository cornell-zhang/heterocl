# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np


def top_syr2k(M=20, N=30, alpha=1.5, beta=1.2, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((N, M), "A")
    B = hcl.placeholder((N, M), "B")
    C = hcl.placeholder((N, N), "C")

    def kernel_syr2k(A, B, C):
        # Irregulax axis access
        with hcl.Stage("loop_1"):
            with hcl.for_(0, N, name="i") as i:
                with hcl.for_(0, i + 1, name="j") as j:
                    C[i][j] *= beta
                with hcl.for_(0, M, name="k") as k:
                    with hcl.for_(0, i + 1, name="j") as j:
                        C[i][j] += A[j][k] * alpha * B[i][k] + B[j][k] * alpha * A[i][k]

    s = hcl.create_schedule([A, B, C], kernel_syr2k)

    #### Apply customizations ####

    #### Apply customizations ####

    return hcl.build(s, target=target)


def syr2k_golden(alpha, beta, M, N, A, B, C, DATA_TYPE):
    dtype = NDATA_TYPE_DICT[DATA_TYPE.lower()]

    for i in range(N):
        for j in range(i + 1):
            C[i][j] *= beta
        for k in range(M):
            for j in range(i + 1):
                C[i][j] += A[j][k] * alpha * B[i][k] + B[j][k] * alpha * A[i][k]
