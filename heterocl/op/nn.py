# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np

dtype = hcl.Float()


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
        raise ValueError("Unknown padding option %s" % padding)
    pad_top = (pad_h + 1) // 2
    pad_left = (pad_w + 1) // 2
    return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left


def pad(data, pad_before, pad_after=None, pad_value=0.0, name="pad"):
    n = len(data.shape)
    pad_after = pad_after if pad_after else pad_before
    if len(pad_before) != n:
        raise ValueError(
            "Input dimension and pad_before dismatch : %d vs %d" % (n, len(pad_before))
        )
    if len(pad_after) != n:
        raise ValueError(
            "Input dimension and pad_after dismatch : %d vs %d" % (n, len(pad_after))
        )
    out_shape = tuple((data.shape[i] + pad_before[i] + pad_after[i]) for i in range(n))

    def _pad(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if pad_before[i] == 0 and pad_after[i] == 0:
                index_tuple.append(indices[i])
            else:
                index_tuple.append(indices[i] - pad_before[i])
                not_zero.append(indices[i] >= pad_before[i])
                not_zero.append(indices[i] < data.shape[i] + pad_before[i])
        if not_zero:
            not_zero = hcl.and_(*not_zero)
            return hcl.select(
                not_zero, data[tuple(index_tuple)], hcl.cast(data.dtype, pad_value)
            )
        return data[tuple(index_tuple)]

    return hcl.compute(out_shape, _pad, name=name, dtype=data.dtype)


def batch_norm(
    data,
    gamma,
    beta,
    moving_mean,
    moving_var,
    axis=1,
    epsilon=10**-7,
    center=1,
    scale=1,
    name="batch_norm",
    dtype=None,
):
    if axis < 0:
        axis = len(data.shape) - 1
    mred = []
    vred = []
    size = 1.0
    for i in range(len(data.shape)):
        if not i == axis:
            mred.append(hcl.reduce_axis(0, data.shape[i], "mred" + str(i)))
            vred.append(hcl.reduce_axis(0, data.shape[i], "vred" + str(i)))
            size = size * data.shape[i]
    new_shape = (data.shape[axis],)

    def insert_axis(axis, red, *indices):
        idx = []
        cur_red = 0
        for i in range(len(data.shape)):
            if i == axis:
                idx.append(indices[0])
            else:
                idx.append(red[cur_red])
                cur_red = cur_red + 1
        return tuple(idx)

    def get_axis(axis, *indices):
        indices = list(indices[0])
        return (indices[axis],)

    if dtype == None:
        dtype = data.dtype
    out = hcl.compute(
        data.shape,
        lambda *x: (data[x] - moving_mean[get_axis(axis, x)])
        / (hcl.sqrt(moving_var[get_axis(axis, x)] + epsilon))
        * gamma[get_axis(axis, x)]
        + beta[get_axis(axis, x)],
        name=name,
        dtype=dtype,
    )
    return out, moving_mean, moving_var


def conv2d_nchw(
    Input,
    Filter,
    strides=[1, 1],
    padding=[0, 0],
    dilation=[1, 1],
    out_dtype=None,
    groups=1,
    name="conv2d",
):
    if out_dtype is None:
        out_dtype = Input.dtype
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

    if groups > 1:
        shape = Filter.shape
        new_shape = (shape[0], groups, shape[2], shape[3])
        Filter = hcl.compute(new_shape, lambda o, i, h, w: Filter[o, 0, h, w])
    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(padding)
    out_channel = num_filter
    out_height = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    if padding != [0, 0]:
        temp = pad(Input, pad_before, pad_after, name=name + "_pad")
    else:
        temp = Input
    if groups > 1:
        rc = hcl.reduce_axis(0, channel / groups, name="rc")
    else:
        rc = hcl.reduce_axis(0, channel, name="rc")
    ry = hcl.reduce_axis(0, kernel_h, name="ry")
    rx = hcl.reduce_axis(0, kernel_w, name="rx")
    if groups > 1:
        return hcl.compute(
            (batch, out_channel, out_height, out_width),
            lambda nn, ff, yy, xx: hcl.sum(
                temp[
                    nn,
                    ff % groups,
                    yy * stride_h + ry * dilation_h,
                    xx * stride_w + rx * dilation_w,
                ]
                * Filter[ff, rc, ry, rx],
                axis=[rc, ry, rx],
                dtype=out_dtype,
            ),
            name=name,
            dtype=out_dtype,
        )
    return hcl.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: hcl.sum(
            temp[
                nn, rc, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w
            ]
            * Filter[ff, rc, ry, rx],
            axis=[rc, ry, rx],
            dtype=out_dtype,
        ),
        name=name,
        dtype=out_dtype,
    )


def conv2d_nhwc(
    Input,
    Filter,
    strides=[1, 1],
    padding=[1, 1],
    dilation=[1, 1],
    out_dtype="float",
    groups=1,
    name="conv2d",
):
    assert isinstance(strides, int) or len(strides) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if out_dtype is None:
        out_dtype = Input.dtype
    if isinstance(strides, int):
        stride_h = stride_w = strides
    else:
        stride_h, stride_w = strides

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_height, in_width, in_channel = Input.shape
    kernel_h, kernel_w, channel, num_filter = Filter.shape

    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(padding)
    out_channel = num_filter
    out_height = (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    temp = pad(Input, pad_before, pad_after, name=name + "_pad")
    rc = hcl.reduce_axis(0, in_channel, name="rc")
    ry = hcl.reduce_axis(0, kernel_h, name="ry")
    rx = hcl.reduce_axis(0, kernel_w, name="rx")
    return hcl.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: hcl.sum(
            temp[
                nn, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, rc
            ]
            * Filter[ry, rx, rc, ff],
            axis=[ry, rx, rc],
            name=name,
            dtype=out_dtype,
        ),
    )


def avg_pool2d_nchw(data, pooling, stride, padding, name="avg_pool2d", dtype=None):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    if dtype == None:
        dtype = data.dtype
    pooling_h, pooling_w = pooling
    stride_h, stride_w = stride
    batch, channel, height, width = data.shape
    if len(padding) == 4:
        pad_top, pad_left, pad_bottom, pad_right = padding
    else:
        pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding)

    if padding != [0, 0]:
        data = pad(data, pad_before, pad_after, pad_value=0.0, name=name + "_pad")
    out_height = (height - pooling_h + pad_top + pad_bottom) // stride_h + 1
    out_width = (width - pooling_w + pad_left + pad_right) // stride_w + 1
    dheight = hcl.reduce_axis(0, pooling_h)
    dwidth = hcl.reduce_axis(0, pooling_w)
    return hcl.compute(
        (batch, channel, out_height, out_width),
        lambda i, c, h, w: (
            hcl.sum(
                data[i, c, h * stride_h + dheight, w * stride_w + dwidth],
                axis=[dheight, dwidth],
                dtype=dtype,
            )
            / (pooling_w * pooling_h)
        ),
        name=name,
        dtype=dtype,
    )


def avg_pool2d_nhwc(
    data, pooling, stride=[1, 1], padding=[0, 0], name="avg_pool2d", dtype=None
):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    if dtype == None:
        dtype = data.dtype
    pooling_h, pooling_w = pooling
    stride_h, stride_w = stride
    batch, height, width, channel = data.shape
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding)
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_bottom, pad_right]
    if padding != [0, 0]:
        data = pad(data, pad_before, pad_after, pad_value=0.0, name=name + "_pad")
    out_height = (height - pooling_h + pad_top + pad_bottom) // stride_h + 1
    out_width = (width - pooling_w + pad_left + pad_right) // stride_w + 1
    dheight = hcl.reduce_axis(0, pooling_h)
    dwidth = hcl.reduce_axis(0, pooling_w)
    return hcl.compute(
        (batch, out_height, out_width, channel),
        lambda i, h, w, c: hcl.sum(
            data[i, h * stride_h + dheight, w * stride_w + dwidth, c],
            axis=[dheight, dwidth],
            dtype=dtype,
        )
        / (pooling_w * pooling_h),
        name=name,
        dtype=dtype,
    )


# def flatten(data, name="flatten", dtype=None):
#     if dtype == None:
#         dtype = data.dtype
#     ishape = data.shape
#     dim = 1
#     for i in range(1, len(ishape)):
#         dim = dim * ishape[i]
#     oshape = (ishape[0], dim)

#     def unwrap(idx, shape):
#         index = []
#         for s in reversed(shape):
#             index.append(idx % s)
#             idx = idx / s
#         return list(reversed(index))

#     return hcl.compute(oshape, lambda i, j: data[tuple([i] + unwrap(j, ishape[1:]))],
#                        name=name, dtype=dtype)


def flatten(data, name="flatten", dtype=None):
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

    return hcl.compute(
        oshape,
        lambda i, j: data[tuple([i] + unwrap(j, ishape[1:]))],
        name=name,
        dtype=dtype,
    )


def flatten_nhwc(data, name="flatten", dtype=None):
    batch, in_height, in_width, channel = data.shape
    out_shape = (batch, in_height * in_width * channel)
    return hcl.compute(
        out_shape,
        lambda i, j: data[
            i, j / (in_width * channel) % in_height, j / channel % in_width, j % channel
        ],
        name=name,
        dtype=dtype,
    )


def dense(data, weight, bias=None, out_dtype=None, name="dense"):
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype == None:
        out_dtype = data.dtype
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = hcl.reduce_axis(0, in_dim)
    matmul = hcl.compute(
        (batch, out_dim),
        lambda i, j: hcl.sum(data[i, k] * weight[j, k], axis=k, dtype=out_dtype),
        name=name + "_matmul",
        dtype=out_dtype,
    )
    if bias is not None:
        matmul = hcl.compute(
            (batch, out_dim),
            lambda i, j: matmul[i, j] + bias[j],
            name=name,
            dtype=out_dtype,
        )
    return matmul
