# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np


def top_mvt(N=40, dtype=hcl.Float(32), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((N, N), "A")
    y1 = hcl.placeholder((N,), "y1")
    y2 = hcl.placeholder((N,), "y2")
    x1 = hcl.placeholder((N,), "x1")
    x2 = hcl.placeholder((N,), "x2")

    def kernel_mvt(A, y1, y2, x1, x2):
        with hcl.Stage("C"):
            with hcl.for_(0, N, name="i") as i:
                with hcl.for_(0, N, name="j") as j:
                    x1[i] = x1[i] + A[i][j] * y1[j]

        with hcl.Stage("D"):
            with hcl.for_(0, N, name="i") as i:
                with hcl.for_(0, N, name="j") as j:
                    x2[i] = x2[i] + A[j][i] * y2[j]

    s = hcl.create_schedule([A, y1, y2, x1, x2], kernel_mvt)

    #### Applying customizations ####

    C = kernel_mvt.C
    D = kernel_mvt.D

    # s[D].reorder(D.axis[1], D.axis[0])
    # s[C].compute_at(s[D], D.axis[1])

    #### Applying customizations ####
    return hcl.build(s, target=target)


def mvt_golden(N, A, y1, y2, x1, x2):
    for i in range(N):
        for j in range(N):
            x1[i] = x1[i] + A[i][j] * y1[j]

    for i in range(N):
        for j in range(N):
            x2[i] = x2[i] + A[j][i] * y2[j]
    return x2


def main(N=40, dtype=hcl.Float(32), target=None):
    f = top_mvt(N, dtype, target)
    A = np.random.randint(10, size=(N, N)).astype(np.float32)
    y1 = np.random.randint(10, size=(N,)).astype(np.float32)
    y2 = np.random.randint(10, size=(N,)).astype(np.float32)
    x1 = np.random.randint(10, size=(N,)).astype(np.float32)
    x2 = np.random.randint(10, size=(N,)).astype(np.float32)
    f(A, y1, y2, x1, x2)
    res_golden = mvt_golden(N, A, y1, y2, x1, x2)
    if np.allclose(x2, res_golden):
        print("pass")
    else:
        print("fail")
        print("output:")
        print(x2)
        print("golden:")
        print(res_golden)


if __name__ == "__main__":
    main()
