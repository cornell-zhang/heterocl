# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np
import math as mt


def top_correlation(M, N, epsilon, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    data = hcl.placeholder((N, M), "data")
    mean = hcl.placeholder((M,), "mean")
    stddev = hcl.placeholder((M,), "stddev")
    corr = hcl.placeholder((M, M), "corr")

    def kernel_correlation(data, mean, stddev, corr):
        k = hcl.reduce_axis(0, N, name="k")
        hcl.update(
            mean, lambda x: hcl.sum(data[k, x], axis=k, dtype=dtype) / N, name="mean"
        )

        m = hcl.reduce_axis(0, N, name="m")
        hcl.update(
            stddev,
            lambda x: hcl.sum(
                (data[m, x] - mean[x]) * (data[m, x] - mean[x]), axis=m, dtype=dtype
            ),
            name="stddev",
        )

        with hcl.Stage("stddevf"):
            with hcl.for_(0, M, name="n") as n:
                stddev[n] = hcl.sqrt(stddev[n] / N * 1.0)
                # This is to avoid a division by zero situation
                with hcl.if_(stddev[n] <= epsilon):
                    stddev[n] = 1.0

        p = hcl.reduce_axis(0, N, name="p")
        cov = hcl.compute(
            (M, M),
            lambda i, j: hcl.sum(
                (data[p, i] - mean[i]) * (data[p, j] - mean[j]), axis=p, dtype=dtype
            )
            / N,
            name="cov",
        )

        with hcl.Stage("corrf"):
            with hcl.for_(0, M, name="q") as q:
                with hcl.for_(0, M, name="r") as r:
                    corr[q, r] = cov[q, r] / (stddev[q] * stddev[r])

    s1 = hcl.create_schedule([data, mean, stddev, corr], kernel_correlation)
    s2 = hcl.create_schedule([data, mean, stddev, corr], kernel_correlation)

    #### Apply customizations ####

    out_mean = kernel_correlation.mean
    out_stddev = kernel_correlation.stddev
    out_stddevf = kernel_correlation.stddevf
    out_cov = kernel_correlation.cov
    out_corrf = kernel_correlation.corrf

    s1[out_mean].compute_at(s1[out_stddev], out_stddev.axis[0])
    s1[out_stddev].compute_at(s1[out_stddevf], out_stddevf.axis[0])
    s1[out_cov].compute_at(s1[out_corrf], out_corrf.axis[0])

    s2[out_mean].compute_at(s2[out_stddev], out_stddev.axis[0])
    s2[out_stddev].compute_at(s2[out_stddevf], out_stddevf.axis[0])
    s2[out_cov].compute_at(s2[out_corrf], out_corrf.axis[1])
    x_outer, x_inner = s2[out_corrf].split(out_corrf.axis[0], factor=10)
    y_outer, y_inner = s2[out_corrf].split(out_corrf.axis[1], factor=10)
    s2[out_corrf].reorder(x_outer, y_outer, x_inner, y_inner)

    return hcl.build(s1, target=target), hcl.build(s2, target=target)


def correlation_golden(M, N, epsilon, mean, stddev, data, corr):
    dtype = float

    float_n = (dtype)(N)

    for j in range(M):
        for i in range(N):
            mean[j] += data[i][j]
        mean[j] /= float_n

    for j in range(M):
        stddev[j] = 0.0
        for i in range(N):
            stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j])
        stddev[j] /= float_n
        stddev[j] = mt.sqrt(stddev[j])

        if stddev[j] <= epsilon:
            stddev[j] = 1.0

    for i in range(N):
        for j in range(M):
            data[i][j] -= mean[j]
            data[i][j] /= mt.sqrt(float_n) * stddev[j]

    for i in range(M - 1):
        corr[i][i] = (dtype)(1)
        for j in range(i + 1, M):
            corr[i][j] = (dtype)(0)
            for k in range(N):
                corr[i][j] += data[k][i] * data[k][j]
            corr[j][i] = corr[i][j]

    corr[M - 1][M - 1] = (dtype)(1)


def main(M=32, N=32, epsilon=0.1, dtype=hcl.Float(32), target=None):
    data = hcl.asarray(
        np.random.randint(10, size=(N, M)).astype(np.float32), hcl.Float(32)
    )
    mean = hcl.asarray(
        np.random.randint(10, size=(M,)).astype(np.float32), hcl.Float(32)
    )
    mean_golden = np.zeros((M,), dtype=np.float32)
    stddev = hcl.asarray(
        np.random.randint(10, size=(M,)).astype(np.float32), hcl.Float(32)
    )
    stddev_golden = np.zeros((M,), dtype=np.float32)
    corr = hcl.asarray(
        np.random.randint(10, size=(M, M)).astype(np.float32), hcl.Float(32)
    )
    corr_golden = np.zeros((M, M), dtype=np.float32)
    f = top_correlation(M, N, epsilon, dtype, target)
    f(data, mean, stddev, corr)
    correlation_golden(M, N, epsilon, mean_golden, stddev_golden, corr_golden)
    if (
        np.allclose(mean.asnumpy(), mean_golden)
        and np.allclose(stddev.asnumpy(), stddev_golden)
        and np.allclose(corr.asnumpy(), corr_golden)
    ):
        print("pass")
    else:
        print("failed")


if __name__ == "__main__":
    main()
