# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import heterocl as hcl


def top_ludcmp(N=40, dtype=hcl.Int(), target=None):
    hcl.init(dtype)

    A = hcl.placeholder((N, N), "A")
    b = hcl.placeholder((N,), "b")
    x = hcl.placeholder((N,), "x")

    # This one is interesting
    # Reusing the code for LU kernel
    # to solve the system of linear equations
    def kernel_ludcmp(A, b, x):
        def solve_lyb(y):
            # Finding solution for LY = b
            with hcl.for_(0, N, name="l1") as i:
                alpha = hcl.scalar(b[i])
                with hcl.for_(0, i, name="l2") as j:
                    alpha.v = alpha.v - A[i][j] * y[j]
                y[i] = alpha.v

        def solve_uxy(y):
            # Finding solution for Ux = y
            with hcl.for_(N - 1, -1, -1, name="l1") as i:
                alpha = hcl.scalar(y[i])
                with hcl.for_(i + 1, N, name="l2") as j:
                    alpha.v = alpha.v - A[i][j] * x[j]
                x[i] = alpha.v / A[i][i]

        def lu():
            with hcl.for_(0, N, name="l1") as i:
                with hcl.for_(0, N, name="l2") as j:
                    with hcl.if_(j < i):
                        w = hcl.scalar(A[i][j])
                        with hcl.for_(0, j, name="l3") as k:
                            w.v = w.v - A[i][k] * A[k][j]
                        A[i][j] = w.v / A[j][j]
                    with hcl.elif_(hcl.and_(j >= i, j < N)):
                        w = hcl.scalar(A[i][j])
                        with hcl.for_(0, i, name="l4") as k:
                            w.v = w.v - A[i][k] * A[k][j]
                        A[i][j] = w.v

        hcl.mutate((1,), lambda x: lu(), name="lu")
        y = hcl.compute((N,), lambda m: 0, "y")

        hcl.mutate((1,), lambda x: solve_lyb(y), name="lyb")
        hcl.mutate((1,), lambda x: solve_uxy(y), name="uxy")

    s = hcl.create_schedule([A, b, x], kernel_ludcmp)

    #### Apply customizations ####

    lu = kernel_ludcmp.lu
    lyb = kernel_ludcmp.lyb
    uxy = kernel_ludcmp.uxy

    ## N Buggy 1
    s[lyb].pipeline(lyb.l1)
    s[uxy].pipeline(uxy.l1)
    s[lu].pipeline(lu.l1)

    #### Apply customizations ####

    return hcl.build(s, target=target)


import numpy as np
import math as mt
from utils.helper import *


def ludcmp_golden(N, A, b, x, DATA_TYPE):
    dtype = NDATA_TYPE_DICT[DATA_TYPE.lower()]

    y = np.zeros((N,), dtype=dtype)

    for i in range(N):
        for j in range(i):
            w = A[i][j]
            for k in range(j):
                w -= A[i][k] * A[k][j]
            A[i][j] = w / A[j][j]
        for j in range(i, N):
            w = A[i][j]
            for k in range(i):
                w -= A[i][k] * A[k][j]

            A[i][j] = w

    for i in range(N):
        w = b[i]
        for j in range(i):
            w -= A[i][j] * y[j]
        y[i] = w

    for i in range(N - 1, -1, -1):
        w = y[i]
        for j in range(i + 1, N):
            w -= A[i][j] * x[j]
        x[i] = w / A[i][i]
