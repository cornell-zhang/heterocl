# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import os


def top_gemver(N=40, alpha=0.1, beta=0.1, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((N, N), "A")
    u1 = hcl.placeholder((N,), "u1")
    u2 = hcl.placeholder((N,), "u2")
    v1 = hcl.placeholder((N,), "v1")
    v2 = hcl.placeholder((N,), "v2")
    x = hcl.placeholder((N,), "x")
    y = hcl.placeholder((N,), "y")
    w = hcl.placeholder((N,), "w")
    z = hcl.placeholder((N,), "z")

    def kernel_gemver(A, u1, u2, v1, v2, x, y, w, z):
        hcl.update(A, lambda i, j: A[i][j] + u1[i] * v1[j] + u2[i] * v2[j], name="A")

        with hcl.Stage("loop_1"):
            with hcl.for_(0, N, name="i") as i:
                with hcl.for_(0, N, name="j") as j:
                    x[i] = x[i] + beta * A[j][i] * y[j]

        hcl.update(x, lambda i: x[i] + z[i], name="x")

        with hcl.Stage("loop_2"):
            with hcl.for_(0, N, name="i") as i:
                with hcl.for_(0, N, name="j") as j:
                    w[i] = w[i] + alpha * A[i][j] * x[j]

    s = hcl.create_schedule([A, u1, u2, v1, v2, x, y, w, z], kernel_gemver)

    #### Apply customizations ####

    A = kernel_gemver.A
    loop_1 = kernel_gemver.loop_1
    x = kernel_gemver.x
    loop_2 = kernel_gemver.loop_2

    # N Buggy 1
    # s[A].reorder(A.axis[1], A.axis[0])
    # s[A].compute_at(s[loop_1], loop_1.axis[0])
    # s[loop_2].reorder(loop_2.axis[1], loop_2.axis[0])
    # s[x].compute_at(s[loop_2], loop_2.axis[1])

    # N Buggy 2
    s[A].reorder(A.axis[1], A.axis[0])
    s[A].compute_at(s[loop_1], loop_1.axis[0])
    s[loop_1].compute_at(s[x], x.axis[0])
    s[loop_2].reorder(loop_2.axis[1], loop_2.axis[0])
    s[x].compute_at(s[loop_2], loop_2.axis[1])

    #### Apply customizations ####

    return hcl.build(s, target=target)


import numpy as np


def gemver_golden(alpha, beta, N, u1, u2, v1, v2, x, y, w, z, A, DATA_TYPE):
    dtype = NDATA_TYPE_DICT[DATA_TYPE.lower()]

    for i in range(N):
        for j in range(N):
            A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]

    for i in range(N):
        for j in range(N):
            x[i] = x[i] + beta * A[j][i] * y[j]

    for i in range(N):
        x[i] = x[i] + z[i]

    for i in range(N):
        for j in range(N):
            w[i] = w[i] + alpha * A[i][j] * x[j]
