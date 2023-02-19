# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import math as mt


def top_nussinov(N, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    seq = hcl.placeholder((N,), "seq")
    table = hcl.placeholder((N, N), "table")

    def kernel_nussinov(seq, table):
        # We will enumerate A G C T (or U in place of T)
        # as 0, 1, 2, and 3 respectively
        # NOTE: Can we use match and max_score function
        #       with scalar arguments? Need to see @Debjit
        # This is completely based on PolyBench implementation
        # Dynamic programming-based analysis
        with hcl.for_(N - 1, -1, -1, name="i") as i:
            with hcl.for_(i + 1, N, name="j") as j:
                with hcl.if_(j - 1 >= 0):
                    with hcl.if_(table[i][j] < table[i][j - 1]):
                        table[i][j] = table[i][j - 1]

                with hcl.if_(i + 1 < N):
                    with hcl.if_(table[i][j] < table[i + 1][j]):
                        table[i][j] = table[i + 1][j]

                with hcl.if_(j - 1 >= 0):
                    with hcl.if_(i + 1 < N):
                        with hcl.if_(i < j - 1):
                            w = hcl.scalar(seq[i] + seq[j])

                            match = hcl.scalar(0.0)
                            with hcl.if_(w.v == 3):
                                match.v = 1.0
                            with hcl.else_():
                                match.v = 0.0

                            s2 = hcl.scalar(table[i + 1][j - 1] + match.v)

                            with hcl.if_(table[i][j] < s2.v):
                                table[i][j] = s2.v

                        with hcl.else_():
                            with hcl.if_(table[i][j] < table[i + 1][j - 1]):
                                table[i][j] = table[i + 1][j - 1]

                with hcl.for_(i + 1, j, name="k") as k:
                    s2 = hcl.scalar(table[i][k] + table[k + 1][j])
                    with hcl.if_(table[i][j] < s2.v):
                        table[i][j] = s2.v

    s = hcl.create_schedule([seq, table], kernel_nussinov)
    return hcl.build(s, target=target)


import numpy as np
import math as mt


def match(b1, b2):
    if b1 + b2 == 3:
        return 1
    else:
        return 0


def max_score(s1, s2):
    if s1 >= s2:
        return s1
    else:
        return s2


def nussinov_golden(N, seq, table, DATA_TYPE):
    dtype = NDATA_TYPE_DICT[DATA_TYPE.lower()]

    for i in range(N - 1, -1, -1):
        for j in range(i + 1, N):
            if j - 1 >= 0:
                table[i][j] = max_score(table[i][j], table[i][j - 1])
            if i + 1 < N:
                table[i][j] = max_score(table[i][j], table[i + 1][j])

            if j - 1 >= 0 and i + 1 < N:
                if i < j - 1:
                    table[i][j] = max_score(
                        table[i][j], table[i + 1][j - 1] + match(seq[i], seq[j])
                    )
                else:
                    table[i][j] = max_score(table[i][j], table[i + 1][j - 1])

            for k in range(i + 1, j):
                table[i][j] = max_score(table[i][j], table[i][k] + table[k + 1][j])


if __name__ == "__main__":
    top_nussinov(32)
