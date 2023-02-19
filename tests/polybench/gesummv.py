# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import os


def top_gesummv(N=30, alpha=0.1, beta=0.1, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((N, N), "A")
    B = hcl.placeholder((N, N), "B")
    x = hcl.placeholder((N,), "x")
    y = hcl.placeholder((N,), "y")

    def kernel_gesummv(A, B, x, y):
        with hcl.Stage("loop_1"):
            with hcl.for_(0, N, name="i") as i:
                with hcl.for_(0, N, name="j") as j:
                    y[i] = y[i] + alpha * A[i][j] * x[j]

        with hcl.Stage("loop_2"):
            with hcl.for_(0, N, name="i") as i:
                with hcl.for_(0, N, name="j") as j:
                    y[i] = y[i] + beta * B[i][j] * x[j]

    s = hcl.create_schedule([A, B, x, y], kernel_gesummv)

    #### Apply customizations ####

    loop_1 = kernel_gesummv.loop_1
    loop_2 = kernel_gesummv.loop_2

    # N Buggy 1
    # s[loop_1].compute_at(s[loop_2], loop_2.axis[1])
    # x_outer, x_inner = s[loop_2].split(loop_2.axis[0], factor=10)
    # y_outer, y_inner = s[loop_2].split(loop_2.axis[1], factor=10)
    # s[loop_2].reorder(x_outer, y_outer, x_inner, y_inner)

    # N Buggy 2
    s[loop_2].compute_at(s[loop_1], loop_1.axis[1])
    x_outer, x_inner = s[loop_1].split(loop_1.axis[0], factor=25)
    y_outer, y_inner = s[loop_1].split(loop_1.axis[1], factor=25)
    s[loop_1].reorder(x_outer, y_outer, x_inner, y_inner)

    #### Apply customizations ####

    return hcl.build(s, target=target)


import numpy as np


def gesummv_golden(alpha, beta, N, A, B, x, y, DATA_TYPE):
    dtype = NDATA_TYPE_DICT[DATA_TYPE.lower()]

    tmp = np.zeros((N,), dtype=dtype)

    for i in range(N):
        for j in range(N):
            tmp[i] = A[i][j] * x[j] + tmp[i]
            y[i] = B[i][j] * x[j] + y[i]

        y[i] = alpha * tmp[i] + beta * y[i]
