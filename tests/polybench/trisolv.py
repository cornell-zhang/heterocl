# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np
import math as mt


def top_trisolv(N, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    L = hcl.placeholder((N, N), "L")
    b = hcl.placeholder((N,), "b")
    x = hcl.placeholder((N,), "x")

    def kernel_trisolv(L, b, x):
        def loop_1():
            with hcl.for_(0, N, name="l1") as i:
                x[i] = b[i]
                with hcl.for_(0, i, name="l2") as j:
                    x[i] = x[i] - L[i][j] * x[j]
                x[i] = x[i] / L[i][i]

        hcl.mutate((1,), lambda x: loop_1(), name="L1")

    s = hcl.create_schedule([L, b, x], kernel_trisolv)

    #### Apply customizations ####

    L1 = kernel_trisolv.L1

    ## N Buggy 1
    s[L1].pipeline(L1.l1)

    #### Apply customizations ####

    return hcl.build(s, target=target)


def trisolv_golden(N, L, b, x, DATA_TYPE):
    for i in range(N):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i][j] * x[j]
        x[i] = x[i] / L[i][i]
