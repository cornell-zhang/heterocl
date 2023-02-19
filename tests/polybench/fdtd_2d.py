# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import math as mt
import os


def top_fdtd_2d(Nx=20, Ny=30, Tmax=20, dtype=hcl.Int(), target=None):
    hcl.init(dtype)

    ex = hcl.placeholder((Nx, Ny), "ex")
    ey = hcl.placeholder((Nx, Ny), "ey")
    hz = hcl.placeholder((Nx, Ny), "hz")
    fict = hcl.placeholder((Tmax,), "fict")

    def kernel_fdtd_2d(ex, ey, hz, fict):
        def update(ex, ey, hz, fict, m):
            const1 = hcl.scalar(0.5, name="const1", dtype=dtype)
            const2 = hcl.scalar(0.7, name="const2", dtype=dtype)

            with hcl.for_(0, Ny, name="L1") as j:
                ey[0][j] = fict[m]

            with hcl.for_(1, Nx, name="L2") as i:
                with hcl.for_(0, Ny, name="L3") as j:
                    ey[i][j] = ey[i][j] - const1.v * (hz[i][j] - hz[i - 1][j])

            with hcl.for_(0, Nx, name="L4") as i:
                with hcl.for_(1, Ny, name="L5") as j:
                    ex[i][j] = ex[i][j] - const1.v * (hz[i][j] - hz[i][j - 1])

            with hcl.for_(0, Nx - 1, name="L6") as i:
                with hcl.for_(0, Ny - 1, name="L7") as j:
                    hz[i][j] = hz[i][j] - const2.v * (
                        ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]
                    )

        hcl.mutate((Tmax,), lambda m: update(ex, ey, hz, fict, m), "main_loop")

    s = hcl.create_schedule([ex, ey, hz, fict], kernel_fdtd_2d)

    #### Apply customization ####

    main_loop = kernel_fdtd_2d.main_loop

    # s[main_loop].unroll(main_loop.L1)
    # s[main_loop].pipeline(main_loop.L2)
    # s[main_loop].unroll(main_loop.L3)
    # fa = s[main_loop].fuse(main_loop.L4, main_loop.L5)
    # s[main_loop].unroll(fa)
    # x_outer, y_outer, x_inner, y_inner = s[main_loop].tile(main_loop.L6, main_loop.L7, 2, 4)
    # s[main_loop].reorder(y_outer, x_outer)

    #### Apply customization ####

    return hcl.build(s, target=target)


import numpy as np
import math as mt


def fdtd_2d_golden(Nx, Ny, Tmax, ex, ey, hz, fict):
    dtype = float

    for t in range(Tmax):
        for j in range(Ny):
            ey[0][j] = fict[t]
        for i in range(1, Nx):
            for j in range(Ny):
                ey[i][j] = ey[i][j] - (dtype)(0.5) * (hz[i][j] - hz[i - 1][j])
        for i in range(Nx):
            for j in range(1, Ny):
                ex[i][j] = ex[i][j] - (dtype)(0.5) * (hz[i][j] - hz[i][j - 1])
        for i in range(Nx - 1):
            for j in range(Ny - 1):
                hz[i][j] = hz[i][j] - (dtype)(0.7) * (
                    ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]
                )


def main(Nx=20, Ny=30, Tmax=20, dtype=hcl.Float(32), target=None):
    f = top_fdtd_2d(Nx, Ny, Tmax, dtype, target)
    ex = np.random.randint(10, size=(Nx, Ny)).astype(np.float32)
    ey = np.random.randint(10, size=(Nx, Ny)).astype(np.float32)
    hz = np.random.randint(10, size=(Nx, Ny)).astype(np.float32)
    fict = np.random.randint(10, size=(Tmax,)).astype(np.float32)
    f(ex, ey, hz, fict)


if __name__ == "__main__":
    main()
