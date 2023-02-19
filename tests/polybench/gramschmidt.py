# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import heterocl as hcl


def top_gramschmidt(M=30, N=20, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    # Rank N matrix
    A = hcl.placeholder((M, N), "A")
    # Orthogonal matrix
    Q = hcl.placeholder((M, N), "Q")
    # Upper triangular matrix
    R = hcl.placeholder((N, N), "R")

    def kernel_gramschmidt(A, Q, R):
        def loop_1():
            with hcl.for_(0, N, name="l1") as k:
                nrm = hcl.scalar(0.0)
                with hcl.for_(0, M, name="l2") as i:
                    nrm.v = nrm.v + A[i][k] * A[i][k]
                R[k][k] = hcl.sqrt(nrm.v * 1.0)
                with hcl.for_(0, M, name="l3") as i:
                    Q[i][k] = A[i][k] / R[k][k]
                with hcl.for_(k + 1, N, name="l4") as j:
                    R[k][j] = 0.0
                    with hcl.for_(0, M, name="l5") as i:
                        R[k][j] = R[k][j] + Q[i][k] * A[i][j]
                    with hcl.for_(0, M, name="l6") as i:
                        A[i][j] = A[i][j] - Q[i][k] * R[k][j]

        hcl.mutate((1,), lambda x: loop_1(), name="L1")

    s = hcl.create_schedule([A, Q, R], kernel_gramschmidt)

    #### Apply customizations ####

    L1 = kernel_gramschmidt.L1
    s[L1].pipeline(L1.l1)

    #### Apply customizations ####

    return hcl.build(s, target=target)


import numpy as np
import math as mt


def gramschmidt_golden(M, N, A, Q, R, DATA_TYPE):
    dtype = NDATA_TYPE_DICT[DATA_TYPE.lower()]

    for k in range(N):
        nrm = (dtype)(0.0)
        for i in range(M):
            nrm += A[i][k] * A[i][k]
        R[k][k] = mt.sqrt(nrm)
        for i in range(M):
            Q[i][k] = A[i][k] / R[k][k]
        for j in range(k + 1, N):
            R[k][j] = (dtype)(0.0)
            for i in range(M):
                R[k][j] += Q[i][k] * A[i][j]
            for i in range(M):
                A[i][j] = A[i][j] - Q[i][k] * R[k][j]
