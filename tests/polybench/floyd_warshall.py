# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import os


def top_floyd_warshall(N=60, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    path = hcl.placeholder((N, N), "path")

    def kernel_floyd_warshall(path):
        def loop_1():
            with hcl.for_(0, N, name="k") as k:
                with hcl.for_(0, N, name="i") as i:
                    with hcl.for_(0, N, name="j") as j:
                        path_ = hcl.scalar(path[i][k] + path[k][j])
                        with hcl.if_(path[i][j] >= path_.v):
                            path[i][j] = path_.v

        hcl.mutate((1,), lambda x: loop_1(), name="L1")

    s = hcl.create_schedule([path], kernel_floyd_warshall)

    #### Apply customizations ####

    L1 = kernel_floyd_warshall.L1

    s[L1].pipeline(L1.k)
    s[L1].unroll(L1.axis[1])
    s[L1].unroll(L1.axis[2])

    #### Apply customizations ####

    return hcl.build(s, target=target)


import numpy as np
import math as mt


def floyd_warshall_golden(N, path, DATA_TYPE):
    dtype = NDATA_TYPE_DICT[DATA_TYPE.lower()]

    for k in range(N):
        for i in range(N):
            for j in range(N):
                path[i][j] = (
                    path[i][j]
                    if path[i][j] < path[i][k] + path[k][j]
                    else path[i][k] + path[k][j]
                )


if __name__ == "__main__":
    top_floyd_warshall()
