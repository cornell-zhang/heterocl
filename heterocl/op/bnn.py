# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=dangerous-default-value

import numpy as np

from ..types import Int, UInt, dtype_to_str
from ..operation import compute, select, cast, reduce_axis, scalar, reducer, sum as sum_
from ..dsl import and_, for_

qtype_bit = UInt(1)


def get_pad_tuple(padding):
    """Common code to get the pad option
    Parameters
    ----------
    padding : Union[int, Tuple[int, ...]]
        Padding size
    Returns
    -------
    pad_top : int
        Padding size on top
    pad_left : int
        Padding size on left
    pad_down : int
        Padding size on down.
    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            pad_h = padding[0] * 2
            pad_w = padding[1] * 2
        elif len(padding) == 4:
            return padding[0], padding[1], padding[2], padding[3]
        else:
            raise ValueError("Size of padding can only be 2 or 4")
    elif isinstance(padding, int):
        pad_h = pad_w = padding * 2
    else:
        raise ValueError(f"Unknown padding option {padding}")
    pad_top = (pad_h + 1) // 2
    pad_left = (pad_w + 1) // 2
    return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left


def if_mac(y, x, in_h, in_w, pad_top, pad_left, pad_down, pad_right):
    return and_(x >= pad_left, x < in_w - pad_right, y >= pad_top, y < in_h - pad_down)


def pad_nchw(data, padding=[0, 0], name="pad", dtype=None):
    assert len(data.shape) == 4, "Only support 4D padding"
    out_dtype = data.dtype if dtype is None else dtype
    batch, channel, in_height, in_width = data.shape
    out_height, out_width = in_height + 2 * padding[0], in_width + 2 * padding[1]
    return compute(
        (batch, channel, out_height, out_width),
        lambda ii, cc, hh, ww: select(
            if_mac(
                hh,
                ww,
                out_height,
                out_width,
                padding[0],
                padding[1],
                padding[0],
                padding[1],
            ),
            data[ii, cc, hh - padding[0], ww - padding[1]],
            cast(out_dtype, 0),
        ),
        dtype=out_dtype,
        name=name,
    )


def pad_nhwc(data, padding=[0, 0], name="pad", dtype=None):
    assert len(data.shape) == 4, "Only support 4D padding"
    if dtype is None:
        dtype = data.dtype
    batch, in_height, in_width, channel = data.shape
    out_height, out_width = in_height + 2 * padding[0], in_width + 2 * padding[1]
    return compute(
        (batch, out_height, out_width, channel),
        lambda ii, hh, ww, cc: select(
            if_mac(
                hh,
                ww,
                out_height,
                out_width,
                padding[0],
                padding[1],
                padding[0],
                padding[1],
            ),
            data[ii, hh - padding[0], ww - padding[1], cc],
            cast(dtype, 0),
        ),
        dtype=dtype,
        name=name,
    )


def conv2d_nchw(
    Input,
    Filter,
    strides=[1, 1],
    padding=[0, 0],
    dilation=[1, 1],
    out_dtype=None,
    name="binary_conv2d",
):
    if out_dtype is None:
        out_dtype = Int()
    assert isinstance(strides, int) or len(strides) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, _, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    # Warning: may have problem with dilated_kernel
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(padding)
    out_channel = num_filter
    out_height = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
    # compute graph
    temp = pad_nchw(Input, padding, name=name + "_pad")
    pad_in_height = in_height + pad_top + pad_down
    pad_in_width = in_width + pad_left + pad_right
    rc = reduce_axis(0, channel, name="rc")
    ry = reduce_axis(0, kernel_h, name="ry")
    rx = reduce_axis(0, kernel_w, name="rx")
    if channel > 1:
        out = compute(
            (batch, out_channel, out_height, out_width),
            lambda nn, ff, yy, xx: sum_(
                select(
                    if_mac(
                        yy * stride_h + ry * dilation_h,
                        xx * stride_w + rx * dilation_w,
                        pad_in_height,
                        pad_in_width,
                        pad_top,
                        pad_left,
                        pad_down,
                        pad_right,
                    ),  # neglect padding pixels in mac
                    cast(
                        out_dtype,
                        (
                            (
                                1
                                - temp[
                                    nn,
                                    rc,
                                    yy * stride_h + ry * dilation_h,
                                    xx * stride_w + rx * dilation_w,
                                ]
                                ^ Filter[ff, rc, ry, rx]
                            )
                            << 1
                        )
                        - 1,
                    ),  # xnor
                    cast(out_dtype, 0),
                ),
                axis=[rc, ry, rx],
                dtype=out_dtype,
                name=name + "_sum",
            ),
            name=name,
            dtype=out_dtype,
        )
    else:  # TODO: otherwise, reuse_at may cause bug
        out = compute(
            (batch, out_channel, out_height, out_width),
            lambda nn, ff, yy, xx: sum_(
                select(
                    if_mac(
                        yy * stride_h + ry * dilation_h,
                        xx * stride_w + rx * dilation_w,
                        pad_in_height,
                        pad_in_width,
                        pad_top,
                        pad_left,
                        pad_down,
                        pad_right,
                    ),  # neglect padding pixels in mac
                    cast(
                        out_dtype,
                        (
                            (
                                1
                                - temp[
                                    nn,
                                    0,
                                    yy * stride_h + ry * dilation_h,
                                    xx * stride_w + rx * dilation_w,
                                ]
                                ^ Filter[ff, 0, ry, rx]
                            )
                            << 1
                        )
                        - 1,
                    ),  # xnor
                    cast(out_dtype, 0),
                ),
                axis=[ry, rx],
                dtype=out_dtype,
                name=name + "_sum",
            ),
            name=name,
            dtype=out_dtype,
        )
    return out


def popcnt(x):
    """
    Reference: https://tlx.github.io/popcount_8hpp_source.html
    Note: Do not use multiplication in popcnt, otherwise it will consume lots
    of hardware resources.
    """
    if x.dtype.bits <= 8:
        x -= (x >> 1) & 0x55
        x = (x & 0x33) + ((x >> 2) & 0x33)
        x = (x + (x >> 4)) & 0x0F
        return x
    if x.dtype.bits <= 16:
        x -= (x >> 1) & 0x5555
        x = (x & 0x3333) + ((x >> 2) & 0x3333)
        x = (x + (x >> 4)) & 0x0F0F
        x += x >> 8
        return x & 0x3F
    if x.dtype.bits <= 32:
        x -= (x >> 1) & 0x55555555
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F
        x += x >> 8
        x += x >> 16
        return x & 0x3F
    if x.dtype.bits <= 64:
        x -= (x >> 1) & 0x5555555555555555
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
        x += x >> 8
        x += x >> 16
        x += x >> 32
        return x & 0x7F
    raise ValueError("Unsupported bitwidth")


def packed_conv2d_nchw(
    Input,
    Filter,
    strides=[1, 1],
    padding=[0, 0],
    dilation=[1, 1],
    out_dtype=None,
    bitwidth=None,
    name="packed_binary_conv2d",
):
    if out_dtype is None:
        out_dtype = Int()
    assert isinstance(strides, int) or len(strides) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if bitwidth is None:
        bitwidth = int(dtype_to_str(Input.dtype).split("int")[-1])
    batch, in_channel, in_height, in_width = Input.shape
    num_filter, filter_channel, kernel_h, kernel_w = Filter.shape
    assert in_channel == filter_channel
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(padding)
    out_channel = num_filter
    out_height = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
    # compute graph
    if list(padding) != [0, 0]:
        temp = pad_nchw(Input, padding, name=name + "_pad", dtype=UInt(bitwidth))
    else:
        temp = Input
    pad_in_height = in_height + pad_top + pad_down
    pad_in_width = in_width + pad_left + pad_right
    rc = reduce_axis(0, in_channel, name=name + "_rc")
    ry = reduce_axis(0, kernel_h, name=name + "_ry")
    rx = reduce_axis(0, kernel_w, name=name + "_rx")
    rb = reduce_axis(0, bitwidth, name=name + "_rb")
    # assert stride_h == 1 and stride_w == 1
    assert dilation_h == 1 and dilation_w == 1
    if bitwidth in {8, 16, 32, 64}:
        out = compute(
            (batch, out_channel, out_height, out_width),
            lambda nn, ff, yy, xx: sum_(
                select(
                    if_mac(
                        yy * stride_h + ry,
                        xx * stride_w + rx,
                        pad_in_height,
                        pad_in_width,
                        pad_top,
                        pad_left,
                        pad_down,
                        pad_right,
                    ),  # neglect padding pixels in mac
                    cast(
                        out_dtype,
                        (
                            bitwidth
                            - (
                                popcnt(
                                    temp[nn, rc, yy * stride_h + ry, xx * stride_w + rx]
                                    ^ Filter[ff, rc, ry, rx]
                                )
                                << 1
                            )
                        ),
                    ),
                    cast(out_dtype, 0),
                ),
                axis=[rc, ry, rx],
                dtype=out_dtype,
                name=name + "_sum",
            ),
            name=name,
            dtype=out_dtype,
        )
    else:
        out = compute(
            (batch, out_channel, out_height, out_width),
            lambda nn, ff, yy, xx: sum_(
                select(
                    if_mac(
                        yy * stride_h + ry,
                        xx * stride_w + rx,
                        pad_in_height,
                        pad_in_width,
                        pad_top,
                        pad_left,
                        pad_down,
                        pad_right,
                    ),  # neglect padding pixels in mac
                    cast(
                        out_dtype,
                        (
                            1
                            - (
                                (
                                    temp[nn, rc, yy * stride_h + ry, xx * stride_w + rx]
                                    ^ Filter[ff, rc, ry, rx]
                                )[rb]
                                * 2
                            )
                        ),
                    ),
                    cast(out_dtype, 0),
                ),
                axis=[rc, ry, rx, rb],
                dtype=out_dtype,
                name=name + "_sum",
            ),
            name=name,
            dtype=out_dtype,
        )
    return out


def packed_conv2d_nhwc(
    Input,
    Filter,
    strides=[1, 1],
    padding=[0, 0],
    dilation=[1, 1],
    out_dtype=None,
    bitwidth=None,
    name="packed_binary_conv2d",
):
    if out_dtype is None:
        out_dtype = Int()
    assert isinstance(strides, int) or len(strides) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if bitwidth is None:
        bitwidth = int(dtype_to_str(Input.dtype).split("int")[-1])
    batch, in_height, in_width, in_channel = Input.shape
    num_filter, kernel_h, kernel_w, filter_channel = Filter.shape
    assert in_channel == filter_channel
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(padding)
    out_channel = num_filter
    out_height = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
    # compute graph
    if padding != [0, 0]:
        temp = pad_nhwc(Input, padding, name=name + "_pad", dtype=UInt(bitwidth))
    else:
        temp = Input
    pad_in_height = in_height + pad_top + pad_down
    pad_in_width = in_width + pad_left + pad_right
    rc = reduce_axis(0, in_channel, name=name + "_rc")
    ry = reduce_axis(0, kernel_h, name=name + "_ry")
    rx = reduce_axis(0, kernel_w, name=name + "_rx")
    # assert stride_h == 1 and stride_w == 1
    assert dilation_h == 1 and dilation_w == 1
    out = compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: sum_(
            select(
                if_mac(
                    yy * stride_h + ry,
                    xx * stride_w + rx,
                    pad_in_height,
                    pad_in_width,
                    pad_top,
                    pad_left,
                    pad_down,
                    pad_right,
                ),  # neglect padding pixels in mac
                cast(
                    out_dtype,
                    (
                        bitwidth
                        - (
                            popcnt(
                                temp[nn, yy * stride_h + ry, xx * stride_w + rx, rc]
                                ^ Filter[ff, ry, rx, rc]
                            )
                            << 1
                        )
                    ),
                ),
                cast(out_dtype, 0),
            ),
            axis=[ry, rx, rc],
            dtype=out_dtype,
            name=name + "_sum",
        ),
        name=name,
        dtype=out_dtype,
    )
    return out


def batch_norm_threshold(data, threshold, name="batch_norm_threshold"):
    return compute(
        data.shape,
        lambda i, c, h, w: select(
            data[i, c, h, w] > threshold[c, h, w],
            cast(qtype_bit, 1),  # quantize
            cast(qtype_bit, 0),
        ),
        name=name,
        dtype=qtype_bit,
    )


def packed_batch_norm_threshold_nhwc(
    data, threshold, name="packed_batch_norm_threshold"
):
    batch, out_height, out_width, channel = data.shape
    bitwidth = channel  # pack channels

    def genpack(i, h, w, c):
        out = scalar(0, name=name + "_pack", dtype=UInt(bitwidth))
        with for_(0, bitwidth) as k:
            out[0][k] = select(
                data[i, h, w, c * bitwidth + k] > threshold[h, w, c * bitwidth + k],
                cast(qtype_bit, 1),
                cast(qtype_bit, 0),
            )
        return out[0]

    return compute(
        (batch, out_height, out_width, channel // bitwidth),
        genpack,
        name=name,
        dtype=UInt(bitwidth),
    )


def max_pool2d_nchw(
    data,
    pooling=[1, 1],
    stride=[1, 1],
    padding=[0, 0],
    name="binary_max_pool2d",
):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    assert data.dtype.bits == 1, "only support binary input data"
    pooling_h, pooling_w = pooling
    stride_h, stride_w = stride
    batch, channel, height, width = data.shape
    if len(padding) == 4:
        pad_top = padding[0]
        pad_left = padding[1]
        pad_bottom = padding[2]
        pad_right = padding[3]
    else:
        pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding)
    if (pad_top, pad_left, pad_bottom, pad_right) != (0, 0, 0, 0):
        raise RuntimeError("Not supported padding")
    out_height = (height - pooling_h + pad_top + pad_bottom) // stride_h + 1
    out_width = (width - pooling_w + pad_left + pad_right) // stride_w + 1
    dheight = reduce_axis(0, pooling_h, "rh")
    dwidth = reduce_axis(0, pooling_w, "rw")
    reducer_ = reducer(0, lambda x, y: x | y, dtype=data.dtype, name=name + "_max")

    return compute(
        (batch, channel, out_height, out_width),
        lambda i, c, h, w: reducer_(
            data[i, c, h * stride_h + dheight, w * stride_w + dwidth],
            axis=[dheight, dwidth],
        ),
        dtype=data.dtype,
        name=name,
    )


def packed_max_pool2d_nhwc(
    data,
    pooling=[1, 1],
    stride=[1, 1],
    padding=[0, 0],
    name="packed_binary_max_pool2d",
):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    assert pooling == [2, 2], "only support [2,2] pooling now"
    assert padding == [0, 0], "only support [0,0] padding now"
    pooling_h, pooling_w = pooling
    stride_h, stride_w = stride
    batch, height, width, channel = data.shape
    bitwidth = int(dtype_to_str(data.dtype).split("int")[-1])
    if len(padding) == 4:
        pad_top, pad_left, pad_bottom, pad_right = padding
    else:
        pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding)
    if (pad_top, pad_left, pad_bottom, pad_right) != (0, 0, 0, 0):
        raise RuntimeError("Not supported padding")
    out_height = (height - pooling_h + pad_top + pad_bottom) // stride_h + 1
    out_width = (width - pooling_w + pad_left + pad_right) // stride_w + 1
    dheight = reduce_axis(0, pooling_h, "rh")
    dwidth = reduce_axis(0, pooling_w, "rw")
    reducer_ = reducer(0, lambda x, y: x | y, dtype=UInt(bitwidth), name=name + "_max")

    maxpool = compute(
        (batch, out_height, out_width, channel),
        lambda i, h, w, c: reducer_(
            data[i, h * stride_h + dheight, w * stride_w + dwidth, c],
            axis=[dheight, dwidth],
        ),
        name=name,
        dtype=UInt(bitwidth),
    )
    return maxpool


def flatten(data, name="flatten"):
    ishape = data.shape
    dim = 1
    for i in range(1, len(ishape)):
        dim = dim * ishape[i]
    oshape = (ishape[0], dim)

    def unwrap(idx, shape):  # channel first
        index = [
            idx % shape[0],
            idx / (shape[0] * shape[1]),
            (idx / shape[0]) % shape[1],
        ]
        return index

    return compute(
        oshape,
        lambda i, j: data[tuple([i] + unwrap(j, ishape[1:]))],
        name=name,
        dtype=data.dtype,
    )


def packed_flatten_nhwc(data, name="packed_flatten"):
    batch, in_height, in_width, channel = data.shape
    out_shape = (batch, in_height * in_width * channel)
    return compute(
        out_shape,
        lambda i, j: data[
            i, j / (in_width * channel) % in_height, j / channel % in_width, j % channel
        ],
        name=name,
        dtype=data.dtype,
    )


def dense(data, weight, bias=None, use_relu=False, dtype=None, name="binary_dense"):
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = reduce_axis(0, in_dim, "k")
    var_w = np.sqrt(2.0 / in_dim)  # predefined constant
    # var_w = 1
    if bias is None:
        matmul = compute(
            (batch, out_dim),
            lambda i, j: sum_(
                cast(dtype, and_(data[i, k] == weight[j, k])),
                axis=k,
                dtype=dtype,
            )
            * 2
            - in_dim,
            name=name + "_matmul",
            dtype=dtype,
        )  # Data type needs to be specified!
    else:
        matmul = compute(
            (batch, out_dim),
            lambda i, j: (
                sum_(
                    cast(bias.dtype, and_(data[i, k] == weight[j, k])),
                    axis=k,
                    dtype=bias.dtype,
                    name=name + "_sum",
                )
                * 2
                - in_dim
            )
            * var_w
            + bias[j],
            name=(name + "_matmul" if use_relu else name),
            dtype=bias.dtype,
        )
    if use_relu:
        matmul = compute(
            (batch, out_dim),
            lambda i, j: select(
                matmul[i, j] > 0, cast(qtype_bit, 1), cast(qtype_bit, 0)
            ),
            name=name,
            dtype=qtype_bit,
        )
    return matmul


def packed_dense(
    data, weight, bias=None, use_relu=False, name="packed_binary_dense", dtype=None
):
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    bitwidth = int(dtype_to_str(data.dtype).split("int")[-1])
    batch, in_dim = data.shape  # in_dim has been packed
    out_dim, _ = weight.shape  # only packed axis 1
    rk = reduce_axis(0, in_dim, name=name + "_rk")
    var_w = np.sqrt(2.0 / in_dim)  # predefined constant
    # var_w = 1
    rb = reduce_axis(0, bitwidth, name=name + "_rb")
    if bias is not None:
        matmul = compute(
            (batch, out_dim),
            lambda i, j: sum_(
                cast(UInt(bitwidth), (data[i, rk] ^ weight[j, rk])[rb]),  # popcount
                axis=[rk, rb],
                name=name + "_popcnt",
                dtype=UInt(bitwidth),
            ),
            name=name + "_matmul",
            dtype=UInt(bitwidth),
        )
    if not use_relu:
        matmul = compute(
            (batch, out_dim),
            lambda i, j: cast(
                bias.dtype, (in_dim * bitwidth - (matmul[i, j] << 1)) * var_w + bias[j]
            ),
            name=name,
            dtype=bias.dtype if dtype is None else dtype,
        )
    else:

        def genpack(i, j):
            out = scalar(0, name=name + "_pack", dtype=data.dtype)
            with for_(0, bitwidth) as k:
                out[0][k] = cast(
                    UInt(1),
                    select(
                        (
                            (in_dim * bitwidth - (matmul[i, j * bitwidth + k] << 1))
                            * var_w
                            + bias[j * bitwidth + k]
                        )
                        > 0,
                        1,
                        0,
                    ),
                )
            return out[0]

        matmul = compute(
            (batch, out_dim // bitwidth),
            genpack,
            name=name,
            dtype=data.dtype if dtype is None else dtype,
        )
    return matmul
