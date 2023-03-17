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


def test_bconv_nhwc_buffer_at():
    bs = 4
    ic, oc = 6, 16
    ih, iw = 8, 8
    kh, kw = 3, 3
    oh, ow = ih - kh + 1, iw - kw + 1

    hcl.init(hcl.UInt(1))
    A = hcl.placeholder((bs, ih, iw, ic))
    F = hcl.placeholder((oc, kh, kw, ic))

    def conv(A, F):
        rc = hcl.reduce_axis(0, ic)
        rh = hcl.reduce_axis(0, kh)
        rw = hcl.reduce_axis(0, kw)
        L = ic * kh * kw
        B = hcl.compute(
            (bs, oh, ow, oc),
            lambda n, h, w, c: L
            - (
                hcl.sum(
                    A[n, h + rh, w + rw, rc] ^ F[c, rh, rw, rc],
                    axis=[rh, rw, rc],
                    dtype=hcl.Int(32),
                )
                << 1
            ),
            name="B",
            dtype=hcl.Int(32),
        )
        return B

    s = hcl.create_schedule([A, F], conv)
    B = conv.B
    buf = s.buffer_at(B, s[B], B.axis[2])
    LB = s.reuse_at(A, s[B], B.axis[1])
    WB = s.reuse_at(LB, s[B], B.axis[2])
    f = hcl.build(s)

    np_A = np.random.randint(0, 2, size=(bs, ih, iw, ic))
    np_B = np.random.randint(0, 2, size=(oc, kh, kw, ic))
    np_C = np.zeros((bs, oh, ow, oc), dtype="int")

    for n in range(0, bs):
        for y in range(0, oh):
            for x in range(0, ow):
                for c in range(0, oc):
                    for rc in range(0, ic):
                        for rh in range(0, kh):
                            for rw in range(0, kw):
                                np_C[n][y][x][c] += 1 - 2 * (
                                    np_A[n][y + rh][x + rw][rc] ^ np_B[c][rh][rw][rc]
                                )

    hcl_A = hcl.asarray(np_A, dtype=hcl.UInt(1))
    hcl_B = hcl.asarray(np_B, dtype=hcl.UInt(1))
    hcl_C = hcl.asarray(np_C, dtype=hcl.Int(32))

    f(hcl_A, hcl_B, hcl_C)

    assert np.array_equal(np_C, hcl_C.asnumpy())


def test_packed_bconv_nhwc_threshold_bufferat():
    # Set up the parameters
    bs = 4
    ic, oc = 6, 16
    ih, iw = 8, 8
    kh, kw = 3, 3
    oh, ow = ih - kh + 1, iw - kw + 1
    packing_factor = 6

    # heterocl kernel
    hcl.init(hcl.UInt(packing_factor))

    def packed_bconv_nhwc(A, F):
        rc = hcl.reduce_axis(0, ic // packing_factor)
        rh = hcl.reduce_axis(0, kh)
        rw = hcl.reduce_axis(0, kw)
        rb = hcl.reduce_axis(0, ic)
        L = ic * kh * kw
        B = hcl.compute(
            (bs, oh, ow, oc),
            lambda n, h, w, c: L
            - (
                hcl.sum(
                    (A[n, h + rh, w + rw, rc] ^ F[c, rh, rw, rc])[rb],
                    axis=[rh, rw, rc, rb],
                    dtype=hcl.Int(32),
                )
                << 1
            ),
            name="B",
            dtype=hcl.Int(32),
        )
        return B

    def packed_batch_norm_threshold_nhwc(data, threshold, name="C"):
        batch, out_height, out_width, channel = data.shape
        bitwidth = channel  # pack channels

        def genpack(i, h, w, c):
            out = hcl.scalar(0, name=name + "_pack", dtype=hcl.UInt(bitwidth))
            with hcl.for_(0, bitwidth) as k:
                out[0][k] = hcl.select(
                    data[i, h, w, c * bitwidth + k] > threshold[h, w, c * bitwidth + k],
                    hcl.cast(hcl.UInt(1), 1),
                    hcl.cast(hcl.UInt(1), 0),
                )
            return out[0]

        return hcl.compute(
            (batch, out_height, out_width, channel // bitwidth),
            genpack,
            name=name,
            dtype=hcl.UInt(bitwidth),
        )

    def two_layer(A, F, X):
        B = packed_bconv_nhwc(A, F)
        C = packed_batch_norm_threshold_nhwc(B, X)
        return B, C

    A = hcl.placeholder((bs, ih, iw, ic // packing_factor))
    F = hcl.placeholder((oc, kh, kw, ic // packing_factor))
    X = hcl.placeholder((oh, ow, oc), dtype=hcl.Int(32))
    s = hcl.create_schedule([A, F, X], two_layer)
    B = two_layer.B
    buf = s.buffer_at(B, s[B], B.axis[2])
    LB = s.reuse_at(A, s[B], B.axis[1])
    WB = s.reuse_at(LB, s[B], B.axis[2])
    f = hcl.build(s)

    np_A = np.random.randint(0, 2, size=(bs, ih, iw, ic))
    np_F = np.random.randint(0, 2, size=(oc, kh, kw, ic))
    np_X = np.random.randint(-9, 9, size=(oh, ow, oc))
    np_B = np.zeros((bs, oh, ow, oc), dtype="int")
    np_C = np.zeros((bs, oh, ow, oc), dtype="int")
    packed_C = np.zeros((bs, oh, ow, 1), dtype="int")

    # convolution
    for n in range(0, bs):
        for y in range(0, oh):
            for x in range(0, ow):
                for c in range(0, oc):
                    for rc in range(0, ic):
                        for rh in range(0, kh):
                            for rw in range(0, kw):
                                np_B[n][y][x][c] += 1 - 2 * (
                                    np_A[n][y + rh][x + rw][rc] ^ np_F[c][rh][rw][rc]
                                )

    # threshold
    for n in range(0, bs):
        for y in range(0, oh):
            for x in range(0, ow):
                for c in range(0, oc):
                    if np_B[n][y][x][c] > np_X[y][x][c]:
                        np_C[n][y][x][c] = 1
                    else:
                        np_C[n][y][x][c] = 0
    # bitpack along channel by oc
    for n in range(0, bs):
        for y in range(0, oh):
            for x in range(0, ow):
                for c in range(0, oc):
                    packed_C[n][y][x][0] |= np_C[n][y][x][c] << c

    packed_A = np.zeros((bs, ih, iw, ic // packing_factor), dtype="int")
    packed_F = np.zeros((oc, kh, kw, ic // packing_factor), dtype="int")
    # pack A
    for n in range(0, bs):
        for y in range(0, ih):
            for x in range(0, iw):
                for c in range(0, ic // packing_factor):
                    for k in range(0, packing_factor):
                        packed_A[n][y][x][c] |= (
                            np_A[n][y][x][c * packing_factor + k] << k
                        )
    # pack F
    for n in range(0, oc):
        for y in range(0, kh):
            for x in range(0, kw):
                for c in range(0, ic // packing_factor):
                    for k in range(0, packing_factor):
                        packed_F[n][y][x][c] |= (
                            np_F[n][y][x][c * packing_factor + k] << k
                        )

    hcl_A = hcl.asarray(packed_A, dtype=hcl.UInt(packing_factor))
    hcl_F = hcl.asarray(packed_F, dtype=hcl.UInt(packing_factor))
    hcl_X = hcl.asarray(np_X, dtype=hcl.Int(32))
    hcl_B = hcl.asarray(np.zeros((bs, oh, ow, oc)), dtype=hcl.Int(32))
    hcl_C = hcl.asarray(np.zeros((bs, oh, ow, 1)), dtype=hcl.UInt(16))

    f(hcl_A, hcl_F, hcl_X, hcl_B, hcl_C)

    assert np.array_equal(np_B, hcl_B.asnumpy())
    assert np.array_equal(packed_C, hcl_C.asnumpy())


def test_bconv2D_nchw_const_tensor():
    bs = 4
    ic, oc = 6, 16
    ih, iw = 8, 8
    kh, kw = 3, 3
    oh, ow = ih - kh + 1, iw - kw + 1
    hcl.init(hcl.UInt(1))
    A = hcl.placeholder((bs, ic, ih, iw))
    F = hcl.placeholder((oc, ic, kh, kw))
    np_B = np.random.randint(0, 2, size=(oc, ic, kh, kw))

    def conv(A):
        rc = hcl.reduce_axis(0, ic)
        rh = hcl.reduce_axis(0, kh)
        rw = hcl.reduce_axis(0, kw)
        L = ic * kh * kw
        F = hcl.const_tensor(np_B, "F", hcl.UInt(1))
        B = hcl.compute(
            (bs, oc, oh, ow),
            lambda n, c, h, w: L
            - (
                hcl.sum(
                    A[n, rc, h + rh, w + rw] ^ F[c, rc, rh, rw],
                    axis=[rc, rh, rw],
                    dtype=hcl.Int(32),
                )
                << 1
            ),
            name="B",
            dtype=hcl.Int(32),
        )
        return B

    s = hcl.create_schedule([A], conv)
    B = conv.B
    LB = s.reuse_at(A, s[B], B.axis[2])
    WB = s.reuse_at(LB, s[B], B.axis[3])
    f = hcl.build(s)

    np_A = np.random.randint(0, 2, size=(bs, ic, ih, iw))
    np_C = np.zeros((bs, oc, oh, ow), dtype="int")

    for n in range(0, bs):
        for c in range(0, oc):
            for y in range(0, oh):
                for x in range(0, ow):
                    for rc in range(0, ic):
                        for rh in range(0, kh):
                            for rw in range(0, kw):
                                np_C[n][c][y][x] += 1 - 2 * (
                                    np_A[n][rc][y + rh][x + rw] ^ np_B[c][rc][rh][rw]
                                )

    hcl_A = hcl.asarray(np_A, dtype=hcl.UInt(1))
    hcl_C = hcl.asarray(np_C, dtype=hcl.Int(32))

    f(hcl_A, hcl_C)

    assert np.array_equal(np_C, hcl_C.asnumpy())
