# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import heterocl.op.bnn as bnn
import numpy as np


def test_packed_bconv_nchw():
    packing_factor = 32
    out_channel = 2
    strides = (1, 1)
    padding = (1, 1)
    in_channel = 1
    bitwidth = 1
    in_dtype = hcl.Float()
    out_dtype = hcl.Float()
    in_shape = (1, 1, 3, 3)  # n, c, h, w
    weight_shape = (2, 1, 3, 3)  # o, i, h, w

    def conv(data, weight):
        data = hcl.compute(
            data.shape,
            lambda *args: hcl.select(data[args] > 0, 1, 0),
            name="data",
            dtype=hcl.UInt(1),
        )
        weight = hcl.compute(
            weight.shape,
            lambda *args: hcl.select(weight[args] > 0, 1, 0),
            name="weight",
            dtype=hcl.UInt(1),
        )
        # pack along channel dimension
        packed_data = hcl.pack(
            data,
            axis=1,
            factor=bitwidth,
            name="conv_packed",
            dtype=hcl.UInt(bitwidth),
        )
        packed_weight = hcl.pack(
            weight,
            axis=1,
            factor=bitwidth,
            name="conv_packed",
            dtype=hcl.UInt(bitwidth),
        )
        return bnn.packed_conv2d_nchw(
            packed_data,
            packed_weight,
            strides=strides,
            padding=padding,
            name="conv_conv2d",
            out_dtype=out_dtype,
        )

    data = hcl.placeholder(in_shape, "data", dtype=in_dtype)
    weight = hcl.placeholder(weight_shape, "weight", dtype=in_dtype)
    s = hcl.create_schedule([data, weight], conv)

    f = hcl.build(s)

    a_np = np.random.randint(0, 10, size=in_shape)
    b_np = np.random.randint(0, 10, size=weight_shape)

    hcl_a = hcl.asarray(a_np, dtype=in_dtype)
    hcl_b = hcl.asarray(b_np, dtype=in_dtype)
    hcl_c = hcl.asarray(np.zeros((1, 2, 3, 3)), dtype=hcl.Float())

    f(hcl_a, hcl_b, hcl_c)

    n, c, h, w = in_shape
    o, i, kh, kw = weight_shape
    # binarize a_np, b_np
    a_np = np.where(a_np > 0, 1, -1)
    b_np = np.where(b_np > 0, 1, -1)
    # pad a_np
    a_np = np.pad(a_np, ((0, 0), (0, 0), (1, 1), (1, 1)), "constant")
    # calculate convolution
    baseline_output = np.zeros((n, o, h, w))
    for i in range(n):
        for j in range(o):
            for k in range(h):
                for l in range(w):
                    for m in range(c):
                        for p in range(kh):
                            for q in range(kw):
                                baseline_output[i][j][k][l] += (
                                    a_np[i][m][k + p][l + q] * b_np[j][m][p][q]
                                )

    assert np.allclose(hcl_c.asnumpy(), baseline_output)


def test_packed_bconv_nchw_with_popcount():
    packing_factor = 32
    out_channel = 2
    strides = (1, 1)
    padding = (1, 1)
    in_channel = 64
    bitwidth = min(in_channel, packing_factor)
    in_dtype = hcl.Float()
    out_dtype = hcl.Float()
    in_shape = (1, 64, 3, 3)  # n, c, h, w
    weight_shape = (64, 64, 3, 3)  # o, i, h, w
    out_shape = (1, 64, 3, 3)

    def conv(data, weight):
        data = hcl.compute(
            data.shape,
            lambda *args: hcl.select(data[args] > 0, 1, 0),
            name="data",
            dtype=hcl.UInt(1),
        )
        weight = hcl.compute(
            weight.shape,
            lambda *args: hcl.select(weight[args] > 0, 1, 0),
            name="weight",
            dtype=hcl.UInt(1),
        )
        # pack along channel dimension
        packed_data = hcl.pack(
            data,
            axis=1,
            factor=bitwidth,
            name="conv_packed",
            dtype=hcl.UInt(bitwidth),
        )
        packed_weight = hcl.pack(
            weight,
            axis=1,
            factor=bitwidth,
            name="conv_packed",
            dtype=hcl.UInt(bitwidth),
        )
        return bnn.packed_conv2d_nchw(
            packed_data,
            packed_weight,
            strides=strides,
            padding=padding,
            name="conv_conv2d",
            out_dtype=out_dtype,
        )

    data = hcl.placeholder(in_shape, "data", dtype=in_dtype)
    weight = hcl.placeholder(weight_shape, "weight", dtype=in_dtype)
    s = hcl.create_schedule([data, weight], conv)

    f = hcl.build(s)

    a_np = np.random.randint(0, 10, in_shape)
    b_np = np.random.randint(0, 10, weight_shape)

    hcl_a = hcl.asarray(a_np, dtype=in_dtype)
    hcl_b = hcl.asarray(b_np, dtype=in_dtype)
    hcl_c = hcl.asarray(np.zeros(out_shape), dtype=hcl.Float())

    f(hcl_a, hcl_b, hcl_c)

    n, c, h, w = in_shape
    o, i, kh, kw = weight_shape
    # binarize a_np, b_np
    a_np = np.where(a_np > 0, 1, -1)
    b_np = np.where(b_np > 0, 1, -1)
    # pad a_np
    a_np = np.pad(a_np, ((0, 0), (0, 0), (1, 1), (1, 1)), "constant")
    # calculate convolution
    baseline_output = np.zeros((n, o, h, w))
    for i in range(n):
        for j in range(o):
            for k in range(h):
                for l in range(w):
                    for m in range(c):
                        for p in range(kh):
                            for q in range(kw):
                                baseline_output[i][j][k][l] += (
                                    a_np[i][m][k + p][l + q] * b_np[j][m][p][q]
                                )

    assert np.allclose(hcl_c.asnumpy(), baseline_output)
