# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np


def top_doitgen(P=12, Q=8, R=10, S=12, dtype=hcl.Float(32), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((R, Q, S), "A")
    x = hcl.placeholder((P, S), "x")

    def kernel_doitgen(A, x):
        sum_ = hcl.compute((P,), lambda x: 0, name="sum_")

        with hcl.for_(0, R, name="r") as r:
            with hcl.for_(0, Q, name="q") as q:
                with hcl.for_(0, P, name="p") as p:
                    sum_[p] = 0
                    with hcl.for_(0, P, name="s") as s:
                        sum_[p] += A[r][q][s] * x[s][p]
                with hcl.for_(0, P, name="p") as p:
                    A[r][q][p] = sum_[p]
        return A

    s = hcl.create_schedule([A, x], kernel_doitgen)

    #### Applying customizations ####

    #### Applying customizations ####

    return hcl.build(s, target=target)


def doitgen_golden(P, Q, R, S, A, x):
    dtype = np.float32
    sum_ = np.zeros((P,), dtype=dtype)
    for r in range(R):
        for q in range(Q):
            for p in range(P):
                sum_[p] = (dtype)(0)
                for s in range(P):
                    sum_[p] += A[r][q][s] * x[s][p]
            for p in range(P):
                A[r][q][p] = sum_[p]


def main(P=12, Q=8, R=10, S=12, dtype=hcl.Float(32), target=None):
    A = np.random.randint(10, size=(R, Q, S)).astype(np.float32)
    x = np.random.randint(10, size=(P, S)).astype(np.float32)
    res = np.zeros(A.shape, dtype=np.float32)
    f = top_doitgen(P, Q, R, S, dtype, target)
    f(A, x, res)
    res_golden = doitgen_golden(P, Q, R, S, A, x)
    if np.allclose(res, res_golden):
        print("pass")
    else:
        print("fail")


if __name__ == "__main__":
    main()
