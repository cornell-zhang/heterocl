# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import os
import numpy as np


def top_symm(M=20, N=30, alpha=1.5, beta=1.2, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((M, M), "A")
    B = hcl.placeholder((M, N), "B")
    C = hcl.placeholder((M, N), "C")

    def kernel_symm(A, B, C):
        # A gemm type approach wont work as A has -999 in the upper
        # triangular part.

        # This implementation follows a verbatim tranlsation from Polybench and
        # LAPACK.
        # http://www.netlib.org/lapack/explore-html/d7/d42/ssymm_8f_source.html
        with hcl.Stage("loop_1"):
            with hcl.for_(0, M, name="i") as i:
                with hcl.for_(0, N, name="j") as j:
                    sum_ = hcl.scalar(0.0)
                    with hcl.for_(0, i, name="k") as k:
                        C[k][j] = C[k][j] + alpha * B[i][j] * A[i][k]
                        sum_.v = sum_.v + B[k][j] * A[i][k]
                    C[i][j] = (
                        beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * sum_.v
                    )

    s = hcl.create_schedule([A, B, C], kernel_symm)

    #### Apply customizations ####

    loop_1 = kernel_symm.loop_1
    # s[L1].pipeline(L1.i)

    #### Apply customizations ####

    return hcl.build(s, target=target)


def symm_golden(alpha, beta, M, N, A, B, C, DATA_TYPE):
    dtype = NDATA_TYPE_DICT[DATA_TYPE.lower()]

    for i in range(M):
        for j in range(N):
            temp2 = (dtype)(0)
            for k in range(i):
                # How the following LoC is derived?
                C[k][j] += alpha * B[i][j] * A[i][k]
                temp2 += B[k][j] * A[i][k]
            C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2


if __name__ == "__main__":
    top_symm()
