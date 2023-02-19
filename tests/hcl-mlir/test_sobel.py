# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl

height = 1280
width = 720

hcl.init(hcl.Float())
A = hcl.placeholder((height, width), "A", dtype=hcl.Float())
Gx = hcl.placeholder((3, 3), "Gx", dtype=hcl.Float())
Gy = hcl.placeholder((3, 3), "Gy", dtype=hcl.Float())


def sobel(A, Gx, Gy):
    r = hcl.reduce_axis(0, 3, "r")
    c = hcl.reduce_axis(0, 3, "c")
    B = hcl.compute(
        (height - 2, width - 2),
        lambda x, y: hcl.sum(A[x + r, y + c] * Gx[r, c], axis=[r, c]),
        name="B",
        dtype=hcl.Float(),
    )
    t = hcl.reduce_axis(0, 3, "t")
    g = hcl.reduce_axis(0, 3, "g")

    C = hcl.compute(
        (height - 2, width - 2),
        lambda x, y: hcl.sum(A[x + t, y + g] * Gy[t, g], axis=[t, g]),
        name="C",
        dtype=hcl.Float(),
    )
    return hcl.compute(
        (height - 2, width - 2),
        lambda x, y: hcl.sqrt(B[x, y] * B[x, y] + C[x, y] * C[x, y]) / 4328.0 * 255.0,
        name="Result",
        dtype=hcl.Float(),
    )


s = hcl.create_schedule([A, Gx, Gy], sobel)
target = hcl.Platform.xilinx_zc706
target.config(compiler="vivado_hls", mode="csyn", project="sobel.prj")
mod = hcl.build(s, target=target)
print(mod.src)
