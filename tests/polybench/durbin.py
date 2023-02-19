# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import math as mt
import os


def top_durbin(N=40, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    r = hcl.placeholder((N,), "r")
    y = hcl.placeholder((N,), "y")

    def kernel_durbin(r, y):
        y[0] = -r[0]
        beta = hcl.scalar(1.0)
        alpha = hcl.scalar(-r[0])

        def update(r, y, k):
            beta.v = (1 - alpha.v * alpha.v) * beta.v
            sum_ = hcl.scalar(0.0)

            z = hcl.compute((N,), lambda m: 0, name="z")

            with hcl.for_(0, k, name="i") as i:
                sum_.v = sum_.v + r[k - i - 1] * y[i]

            alpha.v = -1.0 * (r[k] + sum_.v) / beta.v

            with hcl.for_(0, k, name="j") as j:
                z[j] = y[j] + alpha.v * y[k - j - 1]

            with hcl.for_(0, k, name="m") as m:
                y[m] = z[m]

            y[k] = alpha.v

        hcl.mutate((N - 1,), lambda k: update(r, y, k + 1), "main_loop")

    s = hcl.create_schedule([r, y], kernel_durbin)

    #### Apply customizations ####

    # N Buggy 1
    # main_loop = kernel_durbin.main_loop
    # s[main_loop].pipeline(main_loop.i)
    # s[main_loop].pipeline(main_loop.j)
    # s[main_loop].pipeline(main_loop.m)

    #### Apply customizations ####

    return hcl.build(s, target=target)


import numpy as np
from utils.helper import *


def durbin_golden(N, r, y, DATA_TYPE):
    dtype = NDATA_TYPE_DICT[DATA_TYPE.lower()]

    z = np.zeros((N,), dtype=dtype)

    y[0] = (dtype)(-1 * r[0])
    beta = (dtype)(1)
    alpha = (dtype)(-1 * r[0])

    for k in range(1, N):
        beta = (dtype)((1 - alpha * alpha) * beta)
        sum_ = (dtype)(0)

        for i in range(k):
            sum_ += r[k - i - 1] * y[i]

        alpha = (dtype)(-1 * (r[k] + sum_) / beta)

        for i in range(k):
            z[i] = y[i] + alpha * y[k - i - 1]

        for i in range(k):
            y[i] = z[i]

        y[k] = alpha
