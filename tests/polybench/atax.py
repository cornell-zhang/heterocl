# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np


def top_atax(M=38, N=42, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((M, N), "A")
    x = hcl.placeholder((N,), "x")
    y = hcl.placeholder((N,), "y")

    def kernel_atax(A, x, y):
        r = hcl.reduce_axis(0, N, "r")
        out_Ax = hcl.compute(
            (M,), lambda m: hcl.sum(A[m, r] * x[r], axis=r, dtype=dtype), name="out_Ax"
        )

        k = hcl.reduce_axis(0, M, "k")
        return hcl.compute(
            y.shape,
            lambda n: hcl.sum(A[k, n] * out_Ax[k], axis=k, dtype=dtype),
            name="y",
        )

    s = hcl.create_schedule([A, x, y], kernel_atax)

    #### Apply customization ####

    out_Ax = kernel_atax.out_Ax
    y = kernel_atax.y

    #### Apply customization ####

    return hcl.build(s, target=target)


def atax_golden(M, N, A, x, y):
    dtype = np.float32
    tmp = np.zeros((M,), dtype=dtype)

    for i in range(N):
        y[i] = 0

    for i in range(M):
        tmp[i] = (dtype)(0.0)
        for j in range(N):
            tmp[i] = tmp[i] + A[i][j] * x[j]

        for j in range(N):
            y[j] = y[j] + A[i][j] * tmp[i]

    return y


def main(M=38, N=42, dtype=hcl.Float(32), target=None):
    A = np.random.randint(10, size=(M, N)).astype(np.float32)
    x = np.random.randint(10, size=(N,)).astype(np.float32)
    y = np.random.randint(10, size=(N,)).astype(np.float32)
    res = np.zeros(y.shape, dtype=np.float32)
    f = top_atax(M, N, dtype, target)
    f(A, x, y, res)
    res_golden = atax_golden(M, N, A, x, y)
    if np.allclose(res, res_golden):
        print("pass")
    else:
        print("fail")


if __name__ == "__main__":
    main()
