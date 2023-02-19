# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

###################################################################
# This implementation is based on the following papaer:
#
# P. Lee and Z. M. Kedem. Automatic data and computation
# decomposition on distributed# memory parallel computers.
# ACM Transactions on Programming Languages and Systems,
# 24(1):1–50, Jan. 2002.
#
# Algorithm of Figure 5
#
###################################################################


import heterocl as hcl
import numpy as np


def top_adi(
    Nx=20,
    Ny=20,
    NT=20,
    Dx=0.1,
    Dy=0.1,
    DT=0.1,
    B1=0.1,
    B2=0.1,
    mu1=0.1,
    mu2=0.1,
    a=0.1,
    b=0.1,
    c=0.1,
    d=0.1,
    e=0.1,
    f=0.1,
    dtype=hcl.Int(),
    target=None,
):
    hcl.init(dtype)
    u = hcl.placeholder((Nx, Ny), "u")
    v = hcl.placeholder((Nx, Ny), "v")
    p = hcl.placeholder((Nx, Ny), "p")
    q = hcl.placeholder((Nx, Ny), "q")

    def kernel_adi(u, v, p, q):
        def sweep(u, v, p, q):
            with hcl.for_(1, Ny - 1, name="L1") as i:
                v[0][i] = hcl.scalar(1.0)
                p[i][0] = hcl.scalar(0.0)
                q[i][0] = v[0][i]
                with hcl.for_(1, Nx - 1, name="L2") as j:
                    p[i][j] = -1.0 * c / (a * p[i][j - 1] + b)
                    q[i][j] = (
                        -1.0 * d * u[j][i - 1]
                        + (1.0 + 2.0 * d) * u[j][i]
                        - f * u[j][i + 1]
                        - a * q[i][j - 1]
                    ) / (a * p[i][j - 1] + b)
                v[Nx - 1][i] = hcl.scalar(1.0)
                with hcl.for_(Nx - 2, 0, -1, name="L3") as j:
                    v[j][i] = p[i][j] * v[j + 1][i] + q[i][j]

            with hcl.for_(1, Nx - 1, name="L4") as i:
                u[i][0] = hcl.scalar(1.0)
                p[i][0] = hcl.scalar(0.0)
                q[i][0] = u[i][0]
                with hcl.for_(1, Ny - 1, name="L5") as j:
                    p[i][j] = -1.0 * f / (d * p[i][j - 1] + e)
                    q[i][j] = (
                        -1.0 * a * v[i - 1][j]
                        + (1.0 + 2 * a) * v[i][j]
                        - c * v[i + 1][j]
                        - d * q[i][j - 1]
                    ) / (d * p[i][j - 1] + e)
                u[i][Ny - 1] = hcl.scalar(1.0)
                with hcl.for_(Ny - 2, 0, -1, name="L6") as j:
                    u[i][j] = p[i][j] * u[i][j + 1] + q[i][j]

        hcl.mutate((NT,), lambda m: sweep(u, v, p, q), "main_loop")

    s = hcl.create_schedule([u, v, p, q], kernel_adi)

    #### Apply customizations ####

    main_loop = kernel_adi.main_loop

    # s[main_loop].pipeline(main_loop.L1)
    # s[main_loop].pipeline(main_loop.L4)

    #### Apply customizations ####

    return hcl.build(s, target=target)


def adi_golden(N, TSTEPS, Dx, Dy, DT, B1, B2, mu1, mu2, a, b, c, d, e, f, u, v, p, q):
    for t in range(TSTEPS):
        ## Column sweep
        for i in range(1, N - 1):
            v[0][i] = 1.0
            p[i][0] = 0.0
            q[i][0] = v[0][i]
            for j in range(1, N - 1):
                p[i][j] = -1.0 * c / (a * p[i][j - 1] + b)
                q[i][j] = (
                    -1.0 * d * u[j][i - 1]
                    + (1.0 + 2.0 * d) * u[j][i]
                    - f * u[j][i + 1]
                    - a * q[i][j - 1]
                ) / (a * p[i][j - 1] + b)
            v[N - 1][i] = 1.0
            for j in range(N - 2, 0, -1):
                v[j][i] = p[i][j] * v[j + 1][i] + q[i][j]

        ## Row sweep
        for i in range(1, N - 1):
            u[i][0] = 1.0
            p[i][0] = 0.0
            q[i][0] = u[i][0]
            for j in range(1, N - 1):
                p[i][j] = -1.0 * f / (d * p[i][j - 1] + e)
                q[i][j] = (
                    -1.0 * a * v[i - 1][j]
                    + (1.0 + 2.0 * a) * v[i][j]
                    - c * v[i + 1][j]
                    - d * q[i][j - 1]
                ) / (d * p[i][j - 1] + e)
            u[i][N - 1] = 1.0
            for j in range(N - 2, 0, -1):
                u[i][j] = p[i][j] * u[i][j + 1] + q[i][j]


def main(
    Nx=20,
    Ny=20,
    NT=20,
    Dx=0.1,
    Dy=0.1,
    DT=0.1,
    B1=0.1,
    B2=0.1,
    mu1=0.1,
    mu2=0.1,
    a=0.1,
    b=0.1,
    c=0.1,
    d=0.1,
    e=0.1,
    f=0.1,
    dtype=hcl.Float(32),
    target=None,
):
    u = np.random.randint(10, size=(Nx, Ny)).astype(np.float32)
    v = np.random.randint(10, size=(Nx, Ny)).astype(np.float32)
    p = np.random.randint(10, size=(Nx, Ny)).astype(np.float32)
    q = np.random.randint(10, size=(Nx, Ny)).astype(np.float32)

    f = top_adi(
        Nx, Ny, NT, Dx, Dy, DT, B1, B2, mu1, mu2, a, b, c, d, e, f, dtype, target
    )

    f(u, v, p, q)


if __name__ == "__main__":
    main()
