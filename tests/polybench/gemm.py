# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np


def top_gemm(P, Q, R, alpha, beta, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((P, Q), "A")
    B = hcl.placeholder((Q, R), "B")
    C = hcl.placeholder((P, R), "C")

    def kernel_gemm(A, B, C):
        r = hcl.reduce_axis(0, Q, "r")
        out_AB = hcl.compute(
            (P, R),
            lambda x, y: hcl.sum(alpha * A[x, r] * B[r, y], axis=r, dtype=dtype),
            name="out_AB",
        )

        return hcl.compute(
            C.shape, lambda x, y: beta * C[x, y] + out_AB[x, y], name="C"
        )

    s1 = hcl.create_schedule([A, B, C], kernel_gemm)
    s2 = hcl.create_schedule([A, B, C], kernel_gemm)
    s3 = hcl.create_schedule([A, B, C], kernel_gemm)

    #### Applying customizations ####

    AB = kernel_gemm.out_AB
    C = kernel_gemm.C

    s1[AB].compute_at(s1[C], C.axis[0])
    s1[AB].parallel(AB.axis[0])
    s1[C].parallel(C.axis[0])

    ab_x_outer, ab_x_inner = s2[AB].split(AB.axis[0], factor=2)
    ab_y_outer, ab_y_inner = s2[AB].split(AB.axis[1], factor=2)
    s2[AB].reorder(ab_x_outer, ab_y_outer, ab_x_inner, ab_y_inner)

    s3[AB].compute_at(s3[C], C.axis[0])
    c_x_outer, c_x_inner = s3[C].split(C.axis[0], factor=2)
    ab_y_outer, ab_y_inner = s3[AB].split(AB.axis[1], factor=2)
    s3[AB].reorder(ab_y_inner, ab_y_outer)

    return (
        hcl.build(s1, target=target),
        hcl.build(s2, target=target),
        hcl.build(s3, target=target),
    )


def main(P=16, Q=22, R=18, alpha=0.1, beta=0.1, dtype=hcl.Float(32), target=None):
    f1, f2, f3 = top_gemm(P, Q, R, alpha, beta, dtype, target)
    A = np.random.randint(10, size=(P, Q)).astype(np.float32)
    B = np.random.randint(10, size=(Q, R)).astype(np.float32)
    C = np.random.randint(10, size=(P, R)).astype(np.float32)
    res1 = np.zeros((P, R), dtype=np.float32)
    res2 = np.zeros((P, R), dtype=np.float32)
    res3 = np.zeros((P, R), dtype=np.float32)
    f1(A, B, C, res1)
    f2(A, B, C, res2)
    f3(A, B, C, res3)
    golden = alpha * np.matmul(A, B) + beta * C
    if (
        np.allclose(golden, res1)
        and np.allclose(res1, res2)
        and np.allclose(res2, res3)
    ):
        print("passed")
    else:
        print("failed")


if __name__ == "__main__":
    main()
