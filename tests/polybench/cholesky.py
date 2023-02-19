# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import math as mt
import numpy as np


def top_cholesky(N, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((N, N), "A")

    def kernel_cholesky(A):
        with hcl.Stage("loop_1"):
            with hcl.for_(0, N, name="i") as i:
                # Case: j < i
                with hcl.for_(0, i, name="j") as j:
                    with hcl.for_(0, j, name="k") as k:
                        A[i][j] = A[i][j] - A[i][k] * A[j][k]
                    A[i][j] = A[i][j] / A[j][j]
                # Case: i == j
                with hcl.Stage("k"):
                    with hcl.for_(0, i, name="k") as k:
                        A[i][i] = A[i][i] - A[i][k] * A[i][k]
                A[i][i] = hcl.sqrt(A[i][i] * 1.0)

    s = hcl.create_schedule([A], kernel_cholesky)
    loop_1 = kernel_cholesky.loop_1

    return hcl.build(s, target=target)


def cholesky_golden(N, A):
    for i in range(N):
        for j in range(i):
            for k in range(j):
                A[i][j] -= A[i][k] * A[j][k]
            A[i][j] /= A[j][j]

        for k in range(i):
            A[i][i] -= A[i][k] * A[i][k]
        A[i][i] = mt.sqrt(A[i][i])


def main(N=2, dtype=hcl.Float(32), target=None):
    # Cholesky input matrix needs to be
    # symmetric and positive-definite.
    A = np.random.randint(10, size=(N, N)).astype(np.float32)
    A_np = np.matmul(A, A.T)
    # A_np is our random Hermitian matrix
    A = hcl.asarray(np.copy(A_np), dtype=hcl.Float(32))
    A_golden = np.copy(A_np)
    cholesky_golden(N, A_golden)
    f = top_cholesky(N, dtype, target)
    f(A)
    if np.allclose(A.asnumpy(), A_golden):
        print("pass")
    else:
        print("failed")


if __name__ == "__main__":
    main()
