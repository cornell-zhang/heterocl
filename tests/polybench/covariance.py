# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np
import math as mt


def top_covariance(M, N, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    data = hcl.placeholder((N, M), "data")
    mean = hcl.placeholder((M,), "mean")
    cov = hcl.placeholder((M, M), "mean")

    def kernel_covariance(data, mean, cov):
        k = hcl.reduce_axis(0, N, name="k")
        hcl.update(
            mean,
            lambda x: hcl.sum(data[k, x], axis=k, dtype=dtype) / N,
            name="out_mean",
        )

        p = hcl.reduce_axis(0, N, name="p")
        hcl.update(
            cov,
            lambda i, j: hcl.sum(
                (data[p, i] - mean[i]) * (data[p, j] - mean[j]), axis=p, dtype=dtype
            )
            / (N - 1),
            name="out_cov",
        )

    s1 = hcl.create_schedule([data, mean, cov], kernel_covariance)
    s2 = hcl.create_schedule([data, mean, cov], kernel_covariance)

    #### Apply customizations ####

    out_mean = kernel_covariance.out_mean
    out_cov = kernel_covariance.out_cov

    # N Buggy 1
    s1[out_mean].compute_at(s1[out_cov], out_cov.axis[0])

    # N Buggy 2
    x_outer, x_inner = s2[out_cov].split(out_cov.axis[0], factor=2)
    y_outer, y_inner = s2[out_cov].split(out_cov.axis[1], factor=2)
    s2[out_cov].reorder(x_outer, y_outer, x_inner, y_inner)

    #### Apply customizations ####

    return hcl.build(s1, target=target), hcl.build(s2, target=target)


def covariance_golden(M, N, mean, data, cov):
    dtype = float

    float_n = (dtype)(N)

    for j in range(M):
        for i in range(N):
            mean[j] += data[i][j]
        mean[j] /= float_n

    for i in range(N):
        for j in range(M):
            data[i][j] -= mean[j]

    for i in range(M):
        for j in range(i, M):
            cov[i][j] = 0.0
            for k in range(N):
                cov[i][j] += data[k][i] * data[k][j]
            cov[i][j] /= float_n - (dtype)(1)
            cov[j][i] = cov[i][j]


def main(M=32, N=32, dtype=hcl.Float(32), target=None):
    data = np.random.randint(10, size=(N, M)).astype(np.float32)
    mean1 = np.random.randint(10, size=(M,)).astype(np.float32)
    mean2 = np.random.randint(10, size=(M,)).astype(np.float32)
    mean_golden = np.zeros(mean1.shape, dtype=np.float32)
    cov1 = np.random.randint(10, size=(M, M)).astype(np.float32)
    cov2 = np.random.randint(10, size=(M, M)).astype(np.float32)
    cov_golden = np.zeros(cov1.shape, dtype=np.float32)
    f1, f2 = top_covariance(M, N, dtype, target)
    f1(data, mean1, cov1)
    f2(data, mean2, cov2)
    covariance_golden(M, N, mean_golden, data, cov_golden)

    if (
        np.allclose(mean_golden, mean1)
        and np.allclose(mean1, mean2)
        and np.allclose(cov_golden, cov1)
        and np.allclose(cov1, cov2)
    ):
        print("pass")
    else:
        print("fail")


if __name__ == "__main__":
    main()
