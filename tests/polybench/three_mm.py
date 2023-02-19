# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np


def top_3mm(P=16, Q=20, R=18, S=24, T=22, dtype=hcl.Float(32), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((P, Q), "A")
    B = hcl.placeholder((Q, R), "B")
    C = hcl.placeholder((R, S), "C")
    D = hcl.placeholder((S, T), "D")
    G = hcl.placeholder((P, T), "G")

    def kernel_3mm(A, B, C, D, G):
        r = hcl.reduce_axis(0, Q, "r")
        out_AB = hcl.compute(
            (P, R),
            lambda x, y: hcl.sum(A[x, r] * B[r, y], axis=r, dtype=dtype),
            name="out_AB",
        )

        k = hcl.reduce_axis(0, S, "k")
        out_CD = hcl.compute(
            (R, T),
            lambda x, y: hcl.sum(C[x, k] * D[k, y], axis=k, dtype=dtype),
            name="out_CD",
        )

        q = hcl.reduce_axis(0, R, "q")
        res = hcl.compute(
            G.shape,
            lambda x, y: hcl.sum(out_AB[x, q] * out_CD[q, y], axis=q, dtype=dtype),
            name="res",
        )
        return res

    s = hcl.create_schedule([A, B, C, D, G], kernel_3mm)

    #### Applying customizations ####
    AB = kernel_3mm.out_AB  ## P x R
    CD = kernel_3mm.out_CD  ## R x T
    G = kernel_3mm.res  ## P x T

    s[CD].reorder(CD.axis[1], CD.axis[0])
    s[AB].compute_at(s[G], G.axis[0])

    return hcl.build(s, target=target)


def three_mm_golden(P, Q, R, S, T, A, B, C, D, G):
    dtype = np.float32
    E = np.zeros((P, R), dtype=dtype)

    for i in range(P):
        for j in range(R):
            for k in range(Q):
                E[i][j] += A[i][k] * B[k][j]

    F = np.zeros((R, T), dtype=dtype)

    for i in range(R):
        for j in range(T):
            for k in range(S):
                F[i][j] += C[i][k] * D[k][j]

    for i in range(P):
        for j in range(T):
            for k in range(R):
                G[i][j] += E[i][k] * F[k][j]
    return G


def main(P=16, Q=20, R=18, S=24, T=22, dtype=hcl.Float(32), target=None):
    f = top_3mm(P, Q, R, S, T, dtype, target)
    A = np.random.randint(10, size=(P, Q)).astype(np.float32)
    B = np.random.randint(10, size=(Q, R)).astype(np.float32)
    C = np.random.randint(10, size=(R, S)).astype(np.float32)
    D = np.random.randint(10, size=(S, T)).astype(np.float32)
    G = np.random.randint(10, size=(P, T)).astype(np.float32)
    res = np.zeros(G.shape, dtype=np.float32)
    f(A, B, C, D, G, res)
    res_golden = three_mm_golden(P, Q, R, S, T, A, B, C, D, G)
    if np.allclose(res, res_golden):
        print("pass")
    else:
        print("fail")


if __name__ == "__main__":
    main()
