# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import math as mt
import os


def top_deriche(
    W=64,
    H=64,
    alpha=0.25,
    k=0.1,
    a1=0.1,
    a2=0.1,
    a3=0.1,
    a4=0.1,
    a5=0.1,
    a6=0.1,
    a7=0.1,
    a8=0.1,
    b1=0.1,
    b2=0.1,
    c1=0.1,
    c2=0.1,
    dtype=hcl.Int(),
    target=None,
):
    hcl.init(dtype)
    ImageIn = hcl.placeholder((W, H), "ImageIn")
    ImageOut = hcl.placeholder((W, H), "ImageOut")

    def kernel_deriche(ImageIn, ImageOut):
        # Implementing y1 via x1, delayed x1, and delayed y1
        # Avoided if-else to ensure compile time know exit conditions
        y1 = hcl.compute((W, H), lambda x, y: 0.0, name="y1")
        y2 = hcl.compute((W, H), lambda x, y: 0.0, name="y2")
        # z1 = hcl.compute((W, H), lambda x, y: 0.0, name="z1")
        # z2 = hcl.compute((W, H), lambda x, y: 0.0, name="z2")

        with hcl.Stage("loop_1"):
            with hcl.for_(0, W, name="i") as i:
                y1_d1 = hcl.scalar(0.0)
                y1_d2 = hcl.scalar(0.0)
                x_d1 = hcl.scalar(0.0)
                with hcl.for_(0, H, name="j") as j:
                    y1[i][j] = (
                        a1 * ImageIn[i][j] + a2 * x_d1.v + b1 * y1_d1.v + b2 * y1_d2.v
                    )
                    x_d1.v = ImageIn[i][j]  ## x(i, j - 1)
                    y1_d2.v = y1_d1.v  ## y1(i, j - 2)
                    y1_d1.v = y1[i][j]  ## y1(i, j - 1)

        with hcl.Stage("loop_2"):
            with hcl.for_(0, W, name="i") as i:
                y2_d1 = hcl.scalar(0.0)
                y2_d2 = hcl.scalar(0.0)
                x_d1 = hcl.scalar(0.0)
                x_d2 = hcl.scalar(0.0)
                with hcl.for_(H - 1, -1, -1, name="j") as j:
                    y2[i][j] = a3 * x_d1.v + a4 * x_d2.v + b1 * y2_d1.v + b2 * y2_d2.v
                    x_d2.v = x_d1.v  ## x(i, j + 2)
                    x_d1.v = ImageIn[i][j]  ## x(i, j + 1)
                    y2_d2.v = y2_d1.v  ## y2(i, j + 2)
                    y2_d1.v = y2[i][j]  ## y2(i, j + 1)

        hcl.update(ImageOut, lambda i, j: c1 * (y1[i][j] + y2[i][j]), name="ImageOut1")
        # Since r is not defined anywhere else other than above, hence
        # to save on-chip space, ImageOut is reused as it has the same
        # dimensions as that of r

        with hcl.Stage("loop_3"):
            with hcl.for_(0, H, name="j") as j:
                ImageOut_d1 = hcl.scalar(0.0)
                y1_d1 = hcl.scalar(0.0)
                y1_d2 = hcl.scalar(0.0)
                with hcl.for_(0, W, name="i") as i:
                    y1[i][j] = (
                        a5 * ImageOut[i][j]
                        + a6 * ImageOut_d1.v
                        + b1 * y1_d1.v
                        + b2 * y1_d2.v
                    )
                    ImageOut_d1.v = ImageOut[i][j]  ## r(i - 1, j)
                    y1_d2.v = y1_d1.v  ## y1(i - 2, j)
                    y1_d1.v = y1[i][j]  ## y1(i - 1, j)

        with hcl.Stage("loop_4"):
            with hcl.for_(0, H, name="j") as j:
                ImageOut_d1 = hcl.scalar(0.0)
                ImageOut_d2 = hcl.scalar(0.0)
                y2_d1 = hcl.scalar(0.0)
                y2_d2 = hcl.scalar(0.0)
                with hcl.for_(W - 1, -1, -1, name="i") as i:
                    y2[i][j] = (
                        a7 * ImageOut_d1.v
                        + a8 * ImageOut_d2.v
                        + b1 * y2_d1.v
                        + b2 * y2_d2.v
                    )
                    ImageOut_d2.v = ImageOut_d1.v  ## r(i + 2, j)
                    ImageOut_d1.v = ImageOut[i][j]  ## r(i + 1, j)
                    y2_d2.v = y2_d1.v  ## y2(i + 2, j)
                    y2_d1.v = y2[i][j]  ## y2(i + 1, j)

        hcl.update(ImageOut, lambda i, j: c2 * (y1[i][j] + y2[i][j]), name="ImageOut2")

    s = hcl.create_schedule([ImageIn, ImageOut], kernel_deriche)

    #### Apply customizations ####

    y1 = kernel_deriche.y1
    y2 = kernel_deriche.y2
    loop_1 = kernel_deriche.loop_1
    loop_2 = kernel_deriche.loop_2
    loop_3 = kernel_deriche.loop_3
    loop_4 = kernel_deriche.loop_4
    ImageOut1 = kernel_deriche.ImageOut1

    # N Buggy 1
    # s[y1].compute_at(s[y2], y2.axis[1])
    # s[y2].compute_at(s[loop_1], loop_1.axis[1])

    # N Buggy 2
    s[y1].compute_at(s[y2], y2.axis[1])
    ## NOTE:The following is not working due to the scalar initialization. compute_at is taking out the
    ##      scalar initialization out of the outermost loop. Ask Sean if that is expected
    # s[loop_1].compute_at(s[loop_2], loop_2.axis[1])

    # Buggy 1
    # s[y1].compute_at(s[y2], y2.axis[1])
    # s[y2].compute_at(s[loop_1], loop_1.axis[1])
    # x1_outer, x1_inner = s[loop_1].split(loop_1.axis[0], factor=10)
    # y1_outer, y1_inner = s[loop_1].split(loop_1.axis[1], factor=10)
    # s[loop_1].reorder(x1_outer, y1_outer, x1_inner, y1_inner)

    #### Apply customizations ####

    return hcl.build(s, target=target)


import numpy as np
import math as mt


def deriche_golden(
    W,
    H,
    alpha,
    k,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    a8,
    b1,
    b2,
    c1,
    c2,
    ImageIn,
    ImageOut,
    DATA_TYPE,
):
    dtype = NDATA_TYPE_DICT[DATA_TYPE.lower()]

    y1 = np.zeros((W, H), dtype=dtype)
    y2 = np.zeros((W, H), dtype=dtype)

    for i in range(W):
        ym1 = (dtype)(0.0)
        ym2 = (dtype)(0.0)
        xm1 = (dtype)(0.0)
        for j in range(H):
            y1[i][j] = a1 * ImageIn[i][j] + a2 * xm1 + b1 * ym1 + b2 * ym2
            xm1 = ImageIn[i][j]
            ym2 = ym1
            ym1 = y1[i][j]

    for i in range(W):
        yp1 = (dtype)(0.0)
        yp2 = (dtype)(0.0)
        xp1 = (dtype)(0.0)
        xp2 = (dtype)(0.0)
        for j in range(H - 1, -1, -1):
            y2[i][j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2
            xp2 = xp1
            xp1 = ImageIn[i][j]
            yp2 = yp1
            yp1 = y2[i][j]

    for i in range(W):
        for j in range(H):
            ImageOut[i][j] = c1 * (y1[i][j] + y2[i][j])

    for j in range(H):
        tm1 = (dtype)(0.0)
        ym1 = (dtype)(0.0)
        ym2 = (dtype)(0.0)
        for i in range(W):
            y1[i][j] = a5 * ImageOut[i][j] + a6 * tm1 + b1 * ym1 + b2 * ym2
            tm1 = ImageOut[i][j]
            ym2 = ym1
            ym1 = y1[i][j]

    for j in range(H):
        tp1 = (dtype)(0.0)
        tp2 = (dtype)(0.0)
        yp1 = (dtype)(0.0)
        yp2 = (dtype)(0.0)
        for i in range(W - 1, -1, -1):
            y2[i][j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2
            tp2 = tp1
            tp1 = ImageOut[i][j]
            yp2 = yp1
            yp1 = y2[i][j]

    for i in range(W):
        for j in range(H):
            ImageOut[i][j] = c2 * (y1[i][j] + y2[i][j])


if __name__ == "__main__":
    top_deriche()
