# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import heterocl as hcl
import numpy as np
import math as mt


def top_lu(N=40, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((N, N), "A")

    def kernel_lu(A):
        def loop_1():
            with hcl.for_(0, N, name="i") as i:
                with hcl.for_(0, i, name="j") as j:
                    with hcl.for_(0, j, name="k") as k:
                        A[i][j] -= A[i][k] * A[k][j]
                    A[i][j] /= A[j][j]

                with hcl.for_(i, N, name="j") as j:
                    with hcl.for_(0, i, name="k") as k:
                        A[i][j] -= A[i][k] * A[k][j]

        hcl.mutate((1,), lambda x: loop_1(), name="L1")

    s = hcl.create_schedule([A], kernel_lu)

    #### Apply customization ####

    L1 = kernel_lu.L1

    ## N Buggy 1

    #### Apply customization ####

    return hcl.build(s, target=target)


def lu_golden(N, A, DATA_TYPE):
    dtype = NDATA_TYPE_DICT[DATA_TYPE.lower()]

    for i in range(N):
        for j in range(i):
            for k in range(j):
                A[i][j] -= A[i][k] * A[k][j]
            A[i][j] /= A[j][j]

        for j in range(i, N):
            for k in range(i):
                A[i][j] -= A[i][k] * A[k][j]
