from collections import OrderedDict
import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np
import hlib
from ..utils import *
from .op import *

dtype = hcl.Float()

sum = hcl.reducer(0, lambda x, y: x + y, dtype)
max = hcl.reducer(-1, lambda x, y: tvm.make.Max(x, y), dtype)
_all = hcl.reducer(True, lambda x, y: x & y, bool)

def simplify(expr):
    return tvm.ir_pass.Simplify(expr) if isinstance(expr, tvm.expr.Expr) else expr


def pad(data, pad_before, pad_after=None, pad_value=0.0, name="pad"):
    n = len(data.shape)
    pad_after = pad_after if pad_after else pad_before
    if len(pad_before) != n:
        raise ValueError(
            "Input dimension and pad_before dismatch : %d vs %d" %
            (n, len(pad_before)))
    if len(pad_after) != n:
        raise ValueError(
            "Input dimension and pad_after dismatch : %d vs %d" %
            (n, len(pad_after)))
    out_shape = tuple(
        tvm.ir_pass.Simplify(
            (data.shape[i] + tvm.const(pad_before[i] + pad_after[i]))
        ) for i in range(n))
    pad_value = pad_value if isinstance(pad_value, tvm.expr.Expr) else tvm.const(pad_value, data.dtype)

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
            not_zero = tvm.all(*not_zero)
            return tvm.select(not_zero, data[tuple(index_tuple)], pad_value)
        return data[tuple(index_tuple)]

    return hcl.compute(out_shape, _pad, name=name)


def relay_pad(data, pad_width, pad_value=0.0,
              pad_mode='constant', frontend='keras'):
    pad_before = []
    pad_after = []
    for padded in pad_width:
        pad_before.append(tvm_to_primitive(padded[0]))
        pad_after.append(tvm_to_primitive(padded[1]))
    return pad(data, pad_before, pad_after, pad_value)


def get_pad_tuple(padding, kernel):
    if isinstance(padding, (tuple, list)):
        pad_h = padding[0] * 2
        pad_w = padding[1] * 2
    elif isinstance(padding, int):
        pad_h = pad_w = padding * 2
    elif padding == "VALID":
        pad_h = 0
        pad_w = 0
    elif padding == "SAME":
        pad_h = kernel[0] - 1
        pad_w = kernel[1] - 1
    else:
        raise ValueError("Unknown padding option %s" % padding)
    pad_top = ((pad_h + 1) // 2)
    pad_left = ((pad_w + 1) // 2)
    return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left


def dilate(data, strides, name="DilatedInput"):
    n = len(data.shape)
    if len(strides) != n:
        raise ValueError(
            "data dimension and strides size dismatch : %d vs %d" %
            (n, len(strides)))

    out_shape = tuple(
        simplify((data.shape[i] - 1) * strides[i] + 1) for i in range(n))

    def _dilate(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if strides[i] != 1:
                index_tuple.append(indices[i] / strides[i])
                not_zero.append((indices[i] % strides[i]).equal(0))
            else:
                index_tuple.append(indices[i])
        if not_zero:
            not_zero = tvm.api.all(*not_zero)
            if not_zero:
                return data(*index_tuple)
            else:
                return hcl.cast(data.dtype, 0.0)
        return data(*index_tuple)
    return hcl.compute(out_shape, _dilate, name=name)


def conv2d_transpose(
        data,
        kernel,
        strides=[1, 1],
        padding=[0, 0],
        out_dtype=None):
    if out_dtype is None:
        out_dtype = data.dtype
    """Implementation of conv2d transpose"""
    batch, in_c, in_h, in_w = data.shape
    _, out_c, filter_h, filter_w = kernel.shape
    stride_h, stride_w = strides
    # dilate stage
    DilatedInput = dilate(
        data, [1, 1, stride_h, stride_w], name='DilatedInput')
    # padding stage
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        padding, (filter_h, filter_w))
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right
    PaddedInput = pad(DilatedInput,
                      [0, 0, bpad_top, bpad_left],
                      [0, 0, bpad_bottom, bpad_right],
                      name='PaddedInput')
    # convolution stage
    out_c = simplify(out_c)
    out_h = simplify((in_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h)
    out_w = simplify((in_w - 1) * stride_w - fpad_left - fpad_right + filter_w)
    dc = hcl.reduce_axis((0, in_c), name='dc')
    dh = hcl.reduce_axis((0, filter_h), name='dh')
    dw = hcl.reduce_axis((0, filter_w), name='dw')

    return hcl.compute(
        (batch, out_c, out_h, out_w),
        lambda b, c, h, w: hcl.sum(
            PaddedInput[b, dc, h + dh, w + dw].astype(out_dtype) *
            kernel[dc, c, filter_h - 1 - dh,
                   filter_w - 1 - dw].astype(out_dtype),
            axis=[dc, dh, dw]), name="conv2d_transpose")


def conv2d(
        Input,
        Filter,
        Bias=None,
        strides=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        groups=1,
        channels=1,
        kernel_size=[1, 1],
        data_layout='NCHW',
        kernel_layout='OIHW',
        out_layout='',
        name="conv2d",
        out_dtype=None,
        frontend='keras'):
    p = []
    s = []
    d = []
    for i in range(len(padding)):
        p.append(tvm_to_primitive(padding[i]))
    for i in range(len(strides)):
        s.append(tvm_to_primitive(strides[i]))
        d.append(tvm_to_primitive(dilation[i]))
    strides = s
    padding = p
    dilation = d
    channels = tvm_to_primitive(channels)
    groups = tvm_to_primitive(groups)
    if out_dtype is None or out_dtype == '':
        out_dtype = Input.dtype
    if data_layout == 'NCHW':
        out = conv2d_nchw(
            Input,
            Filter,
            strides,
            padding,
            dilation,
            name='conv2d',
            groups=groups,
            out_dtype=out_dtype)
    elif data_layout == 'NHWC':
        out = conv2d_nhwc(
            Input,
            Filter,
            strides,
            padding,
            dilation,
            name='conv2d',
            groups=groups,
            out_dtype=out_dtype)
    elif data_layout == 'HWCN':
        out = conv2d_hwcn(
            Input,
            Filter,
            strides,
            padding,
            dilation,
            name='conv2d',
            groups=groups,
            out_dtype=out_dtype)
    else:
        raise ValueError("not support this layout {} yet".format(data_layout))
    return out


def conv2d_nhwc(
        Input,
        Filter,
        strides=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        out_dtype='float',
        groups=1,
        name='conv2d'):
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
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify(
        (in_height -
         dilated_kernel_h +
         pad_top +
         pad_down) //
        stride_h +
        1)
    out_width = simplify(
        (in_width -
         dilated_kernel_w +
         pad_left +
         pad_right) //
        stride_w +
        1)
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    temp = pad(Input, pad_before, pad_after, name="temp")
    rc = hcl.reduce_axis(0, in_channel, name='rc')
    ry = hcl.reduce_axis(0, kernel_h, name='ry')
    rx = hcl.reduce_axis(0, kernel_w, name='rx')
    return hcl.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: hcl.sum(
            temp[nn, yy * stride_h + ry * dilation_h,
                 xx * stride_w + rx * dilation_w, rc] *
            Filter[ry, rx, rc, ff], axis=[ry, rx, rc],
            name=name, dtype=out_dtype))


def conv2d_nchw(
        Input,
        Filter,
        strides=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        out_dtype=None,
        groups=1,
        name='conv2d'):
    if out_dtype is None or out_dtype == '':
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
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify(
        (in_height -
         dilated_kernel_h +
         pad_top +
         pad_down) //
        stride_h +
        1)
    out_width = simplify(
        (in_width -
         dilated_kernel_w +
         pad_left +
         pad_right) //
        stride_w +
        1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = pad(Input, pad_before, pad_after, name="pad_temp")
    if groups > 1:
        rc = hcl.reduce_axis(0, channel / groups, name='rc')
    else:
        rc = hcl.reduce_axis(0, channel, name='rc')
    ry = hcl.reduce_axis(0, kernel_h, name='ry')
    rx = hcl.reduce_axis(0, kernel_w, name='rx')
    if groups > 1:
        return hcl.compute(
            (batch, out_channel, out_height, out_width),
            lambda nn, ff, yy, xx: hcl.sum(
                temp[nn, ff % groups, yy * stride_h + ry * dilation_h,
                     xx * stride_w + rx * dilation_w] *
                Filter[ff, rc, ry, rx],
                axis=[rc, ry, rx], dtype=out_dtype), name=name, dtype=out_dtype)
    return hcl.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: hcl.sum(
            temp[nn, rc, yy * stride_h + ry * dilation_h,
                 xx * stride_w + rx * dilation_w] *
            Filter[ff, rc, ry, rx],
            axis=[rc, ry, rx], dtype=out_dtype), name=name, dtype=out_dtype)


def conv2d_hwcn(
        Input,
        Filter,
        strides=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        out_dtype=None,
        groups=1,
        name='conv2d'):
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

    in_height, in_width, in_channel, batch = Input.shape
    kernel_h, kernel_w, channel, num_filter = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify(
        (in_height -
         dilated_kernel_h +
         pad_top +
         pad_down) //
        stride_h +
        1)
    out_width = simplify(
        (in_width -
         dilated_kernel_w +
         pad_left +
         pad_right) //
        stride_w +
        1)
    pad_before = [pad_top, pad_left, 0, 0]
    pad_after = [pad_down, pad_right, 0, 0]
    temp = pad(Input, pad_before, pad_after, name="temp")
    rc = hcl.reduce_axis(0, in_channel, name='rc')
    ry = hcl.reduce_axis(0, kernel_h, name='ry')
    rx = hcl.reduce_axis(0, kernel_w, name='rx')
    return tvm.compute(
        (out_height, out_width, out_channel, batch),
        lambda yy, xx, ff, nn: hcl.sum(
            temp[yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w,
                 rc, nn] * Filter[ry, rx, rc, ff], axis=[ry, rx, rc],
            dtype=out_dtype),
        name=name)


def conv2d_nchw_old(
        Input,
        Filter,
        name="conv2d",
        stride=[1, 1],
        padding=[[0, 0], [0, 0]]):
    out_dtype = Input.dtype
    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    stride_h, stride_w = stride
    [pad_top, pad_left], [pad_down, pad_right] = padding
    # compute the output shape
    out_channel = num_filter
    out_height = simplify(
        (in_height -
         kernel_h +
         pad_top +
         pad_down) //
        stride_h +
        1)
    out_width = simplify(
        (in_width -
         kernel_w +
         pad_left +
         pad_right) //
        stride_w +
        1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    if padding != [[0, 0], [0, 0]]:
        Input = pad(Input, pad_before, pad_after)
    rc = hcl.reduce_axis(0, in_channel)
    ry = hcl.reduce_axis(0, kernel_h)
    rx = hcl.reduce_axis(0, kernel_w)
    return hcl.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: sum(
            Input[nn, rc, yy * stride_h + ry, xx * stride_w + rx] *
            Filter[ff, rc, ry, rx],
            axis=[rc, ry, rx]),
        name=name,
        attrs=OrderedDict([
            ('p', kernel_h),
            ('q', kernel_w),
            ('in_num', in_channel),
            ('out_num', out_channel),
            ('out_img_w', out_width),
            ('out_img_h', out_height),
            ('cin_dtype', tvm.make.StringImm(Input.dtype)),
            ('filter_dtype', tvm.make.StringImm(Filter.dtype)),
            ('app_name', tvm.make.StringImm('cnn'))]))


def dense(data, weight, units=None, out_dtype='', bias=None, name="dense"):
    assert len(
        data.shape) == 2 and len(
        weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = hcl.reduce_axis(0, in_dim)
    attrs = OrderedDict([
        ('k', in_dim),
        ('j', out_dim),
        ('i', batch),
        ('app_name', tvm.make.StringImm('mm'))])
    matmul = hcl.compute((batch, out_dim), lambda i, j: sum(
        data[i, k] * weight[j, k], axis=k), name, attrs=attrs)
    if bias is not None:
        matmul = hcl.compute(
            (batch, out_dim),
            lambda i, j: matmul[i, j] + bias[j],
            name=name,
            attrs=attrs)
    return matmul


def bias_add(data, bias, axis=-1, name='bias_add'):
    def _broadcast(shape, *indices):
        axes = []
        indices = indices[0]
        for i in range(len(shape)):
            if shape[i] == 1:
                axes.append(0)
            else:
                axes.append(indices[i])
        return tuple(axes)
    data_len = len(data.shape)
    if axis < 0:
        axis += data_len
    num_newaxis = data_len - axis - 1
    bias = expand_dims(bias, len(bias.shape), num_newaxis)
    bias = expand_dims(bias, 0, axis)
    return hcl.compute(
        data.shape, lambda *x: data[x] + bias[_broadcast(bias.shape, x)], name=name)


def squeeze(data, axis=None, name='squeeze'):
    if axis is None:
        axis = []
        for i in range(len(data.shape)):
            if data.shape[i] == 1:
                axis.append(i)
    else:
        axis = [tvm_to_primitive(x) for x in axis]
    l_orig = len(data.shape)
    new_shape = []
    for i in range(len(data.shape)):
        if not i in axis:
            new_shape.append(data.shape[i])

    def _ind(axis, l_orig, *indices):
        indices = indices[0]
        new_shape = []
        idx = 0
        for i in range(l_orig):
            if i not in axis:
                new_shape.append(indices[idx])
                idx = idx + 1
            else:
                new_shape.append(0)
        return tuple(new_shape)
    return hcl.compute(
        tuple(new_shape), lambda *x: data[_ind(axis, l_orig, x)], name=name)


def split(data, indices_or_sections, axis=0, name='split'):
    def split_idx(start, axis, *indices):
        indices = indices[0]
        new_ind = []
        for i in range(len(indices)):
            if i == axis:
                new_ind.append(start + indices[i])
            else:
                new_ind.append(indices[i])
        return tuple(new_ind)
    try:
        if not hasattr(indices_or_sections, "value"):
            _list = []
            for section in indices_or_sections:
                _list.append(tvm_to_primitive(section))
            indices_or_sections = _list
    except BaseException:
        _list = []
        if isinstance(indices_or_sections, int):
            pass
        else:
            for section in indices_or_sections:
                _list.append(tvm_to_primitive(section))
            indices_or_sections = _list
    if not isinstance(indices_or_sections, list):
        if hasattr(indices_or_sections, "value"):
            indices_or_sections = indices_or_sections.value
        assert (axis >= 0 & axis < len(data.shape)
                ), "axis not in bounds of shape"
        assert(
            data.shape[axis] %
            indices_or_sections == 0), "indices doesn't divide equally"
        new_shape = list(data.shape)
        intval = new_shape[axis] // indices_or_sections
        new_shape[axis] = intval
        out = []
        for i in range(indices_or_sections):
            out.append(hcl.compute(
                tuple(new_shape), lambda *x: data[split_idx(i * intval, axis, x)], name=name))
    else:
        new_shape = []
        for s in range(len(indices_or_sections)):
            if hasattr(indices_or_sections[s], "value"):
                indices_or_sections[s] = indices_or_sections[s].value
        for i in range(len(indices_or_sections) + 1):
            new_shape.append(list(data.shape))
        start = 0
        axis_width = []
        for i in range(len(indices_or_sections)):
            ax = indices_or_sections[i]
            new_shape[i][axis] = ax - start
            axis_width.append(start)
            start = ax
        ax = new_shape[-1][axis]
        new_shape[-1][axis] = ax - start
        axis_width.append(start)
        out = []
        for i in range(len(indices_or_sections) + 1):
            out.append(hcl.compute(tuple(
                new_shape[i]), lambda *x: data[split_idx(axis_width[i], axis, x)], name=name))
    return tuple(out)


def concatenate(*data_tup, axis=1, name='concatenate', frontend='keras'):
    idx_start = [0]
    axis_len = 0
    for i in range(len(data_tup)):
        idx_start.append(idx_start[i] + (data_tup[i]).shape[axis])
        axis_len = axis_len + (data_tup[i]).shape[axis]

    new_shape = list(data_tup[0].shape)
    new_shape[axis] = axis_len
    C = hcl.placeholder(tuple(new_shape))

    def concat(data, offset, *indices):
        orig_idx = list(indices[0])
        idx = list(indices[0])
        idx[axis] = idx[axis] + offset
        orig_idx = tuple(orig_idx)
        idx = tuple(idx)
        C[idx] = data[orig_idx]
    for i in range(len(data_tup)):
        hcl.mutate(data_tup[i].shape,
                   lambda *x: concat(data_tup[i], idx_start[i], x),
                   name=name)
    return C


def reduce_mult(l):
    result = 1
    for item in l:
        result = result * item
    return result


def reshape(data, newshape, name='reshape'):
    new_shape = []
    for i in range(len(newshape)):
        new_shape.append(tvm_to_primitive(newshape[i]))
    res_shape = []
    cur_shape = list(data.shape)
    idx = 0
    idx_n1 = -1
    for _ in range(len(new_shape)):
        new_idx = new_shape[idx]
        assert(new_idx > -5), "idx has to be greater than -5"
        if new_idx > 0:
            res_shape.append(new_idx)
        elif new_idx == 0:
            res_shape.append(cur_shape[idx])
        elif new_idx == -1:
            if not idx_n1 == -1:
                raise ValueError("no more than one -1 is allowed in newshape")
            idx_n1 = idx
        elif new_idx == -2:
            res_shape.extend(cur_shape[idx:])
        elif new_idx == -3:
            res_shape.append(cur_shape[idx] + cur_shape[idx + 1])
            idx = idx + 1
        elif new_idx == -4:
            assert False, "not implemented yet"
        idx = idx + 1
    if not idx_n1 == -1:
        res_shape.insert(idx_n1, reduce_mult(cur_shape) // reduce_mult(res_shape))
    assert(reduce_mult(cur_shape) == reduce_mult(res_shape)), "shape must contain same total product"
    cur_order = [1]
    res_order = [1]
    c_ord = len(cur_shape) - 1
    for i in range(len(cur_shape) - 1):
        cur_order.append(cur_shape[c_ord - i] * cur_order[i])
    r_ord = len(res_shape) - 1
    for i in range(len(res_shape) - 1):
        res_order.append(res_shape[r_ord - i] * res_order[i])

    def _reshape_idx(*indices):
        indices = indices[0]
        elm_idx = 0
        data_idx = []
        for i in range(len(indices)):
            elm_idx = indices[r_ord - i] * res_order[i] + elm_idx
        for i in range(len(cur_order)):
            data_idx.append((elm_idx // (cur_order[c_ord - i])) % cur_shape[i])
        return tuple(data_idx)
    return hcl.compute(
        tuple(res_shape), lambda *x: data[_reshape_idx(x)], name=name)


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
        name="batch_norm"):
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
    out = hcl.compute(data.shape, lambda *x: (data[x] - moving_mean[get_axis(axis, x)]) /
                    (hcl.sqrt(moving_var[get_axis(axis, x)] + epsilon)) * gamma[get_axis(axis, x)]
                    + beta[get_axis(axis, x)], name=name, dtype=data.dtype)
    return out, moving_mean, moving_var


def batch_matmul(x, y, name="batch_matmul"):
    out_shape = (x.shape[0], x.shape[1], y.shape[2])
    k = hcl.reduce_axis(0, x.shape[2], "k")
    return hcl.compute(out_shape, lambda b, m, n: hcl.sum(x[b,m,k] * y[b,n,k], axis = [k]), name=name, dtype = x.dtype)


def dropout(data, rate=0.5):
    data = hcl.compute(data.shape, lambda *x: data[x])
    mask = hcl.compute(data.shape, lambda *x: hcl.cast(dtype, 1))
    return data, mask


def max_pool(data, kernel, stride, padding=[[0, 0], [0, 0]], name="max_pool2d"):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    kernel_height, kernel_width = kernel
    stride_height, stride_width = stride
    batch, channel, height, width = data.shape
    [pad_top, pad_left], [pad_down, pad_right] = padding
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    if padding != [[0, 0], [0, 0]]:
        data = pad(data, pad_before, pad_after, pad_value=tvm.min_value(data.dtype))
    out_height = simplify(
        (height -
         kernel_height +
         pad_top +
         pad_down) //
        stride_height +
        1)
    out_width = simplify(
        (width -
         kernel_width +
         pad_left +
         pad_right) //
        stride_width +
        1)
    dheight = hcl.reduce_axis(0, kernel_height)
    dwidth = hcl.reduce_axis(0, kernel_width)

    return hcl.compute(
        (batch, channel, out_height, out_width),
        lambda i, c, h, w: max(data[i, c, h *
                                    stride_height +
                                    dheight, w *
                                    stride_width +
                                    dwidth], axis=[dheight, dwidth]),
        name=name,
        attrs=OrderedDict([
            ('out_img_w', out_width),
            ('out_img_h', out_height),
            ('in_num', channel),
            ('kernel_h', kernel[1]),
            ('kernel_w', kernel[0]),
            ('stride_h', stride[1]),
            ('stride_w', stride[0]),
            ('app_name', tvm.make.StringImm('max_pool'))]))


def max_pool2d(
        data,
        pool_size=[1, 1],
        strides=[1, 1],
        padding=[0, 0],
        layout='NCHW',
        name='max_pool2d'):
    pooling = []
    stride = []
    pad = []
    for i in range(len(pool_size)):
        pooling.append(tvm_to_primitive(pool_size[i]))
        stride.append(tvm_to_primitive(strides[i]))
    for i in range(len(padding)):
        pad.append(tvm_to_primitive(padding[i]))
    if layout == 'NCHW':
        out = max_pool2d_nchw(data, pooling, stride, pad, name)
    elif layout == 'NHWC':
        out = max_pool2d_nhwc(data, pooling, stride, pad, name)
    else:
        raise ValueError("not support this layout {} yet".format(layout))
    return out


def max_pool2d_nchw(data, pooling, stride, padding, name='max_pool2d'):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    max = hcl.reducer(
        tvm.min_value(data.dtype),
        lambda x, y: tvm.make.Max(x, y),
        data.dtype)
    pooling_h, pooling_w = pooling
    stride_h, stride_w = stride
    batch, channel, height, width = data.shape
    if len(padding) == 4:
        pad_top = padding[0]
        pad_left = padding[1]
        pad_bottom = padding[2]
        pad_right = padding[3]
    else:
        pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (pooling_h, pooling_w))
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_bottom, pad_right]
    data = pad(data, pad_before, pad_after, pad_value=tvm.min_value(data.dtype))
    out_height = simplify(
        (height - pooling_h + pad_top + pad_bottom) // stride_h + 1)
    out_width = simplify(
        (width - pooling_w + pad_left + pad_right) // stride_w + 1)
    dheight = hcl.reduce_axis(0, pooling_h)
    dwidth = hcl.reduce_axis(0, pooling_w)
    return hcl.compute(
        (batch, channel, out_height, out_width),
        lambda i, c, h, w: max(data[i, c, h *
                                    stride_h +
                                    dheight, w *
                                    stride_w +
                                    dwidth], axis=[dheight, dwidth]),
        name=name, dtype=data.dtype,
        attrs=OrderedDict([
            ('out_img_w', out_width),
            ('out_img_h', out_height),
            ('in_num', channel),
            ('kernel_h', pooling[1]),
            ('kernel_w', pooling[0]),
            ('stride_h', stride[1]),
            ('stride_w', stride[0]),
            ('app_name', tvm.make.StringImm('max_pool'))]))


def max_pool2d_nhwc(
        data,
        pooling,
        stride=[1, 1],
        padding=[0, 0],
        name='max_pool2d'):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    max = hcl.reducer(
        tvm.min_value(
            data.dtype), lambda x, y: tvm.make.Max(
            x, y), data.dtype)
    pooling_h, pooling_w = pooling
    stride_h, stride_w = stride
    batch, height, width, channel = data.shape
    if len(padding) == 4:
        pad_top = padding[0]
        pad_left = padding[1]
        pad_bottom = padding[2]
        pad_right = padding[3]
    else:
        pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (pooling_h, pooling_w))
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_bottom, pad_right, 0]
    data = pad(
        data,
        pad_before,
        pad_after,
        pad_value=tvm.min_value(
            data.dtype))
    out_height = simplify(
        (height - pooling_h + pad_top + pad_bottom) // stride_h + 1)
    out_width = simplify(
        (width - pooling_w + pad_left + pad_right) // stride_w + 1)
    dheight = hcl.reduce_axis(0, pooling_h)
    dwidth = hcl.reduce_axis(0, pooling_w)
    return hcl.compute(
        (batch, out_height, out_width, channel),
        lambda i, h, w, c: max(data[i, h *
                                    stride_h +
                                    dheight, w *
                                    stride_w +
                                    dwidth, c], axis=[dheight, dwidth]),
        name=name,
        attrs=OrderedDict([
            ('out_img_w', out_width),
            ('out_img_h', out_height),
            ('in_num', channel),
            ('kernel_h', pooling[1]),
            ('kernel_w', pooling[0]),
            ('stride_h', stride[1]),
            ('stride_w', stride[0]),
            ('app_name', tvm.make.StringImm('max_pool'))]))


def avg_pool2d(
        data,
        pool_size=[1, 1],
        strides=[1, 1],
        padding=[0, 0],
        layout='NCHW',
        name='avg_pool2d'):
    pooling = []
    stride = []
    pad = []
    for i in range(len(pool_size)):
        pooling.append(tvm_to_primitive(pool_size[i]))
        stride.append(tvm_to_primitive(strides[i]))
        pad.append(tvm_to_primitive(padding[i]))
    if layout == 'NCHW':
        out = avg_pool2d_nchw(data, pooling, stride, pad, name)
    elif layout == 'NHWC':
        out = avg_pool2d_nhwc(data, pooling, stride, pad, name)
    else:
        raise ValueError("not support this layout {} yet".format(layout))
    return out


def avg_pool2d_nchw(data, pooling, stride, padding, name='avg_pool2d'):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    pooling_h, pooling_w = pooling
    stride_h, stride_w = stride
    batch, channel, height, width = data.shape
    if len(padding) == 4:
        pad_top, pad_left, pad_bottom, pad_right = padding
    else:
        pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(
            padding, (pooling_h, pooling_w))
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_bottom, pad_right]
    data = pad(
        data,
        pad_before,
        pad_after,
        pad_value=tvm.const(
            0.0,
            data.dtype))
    out_height = simplify(
        (height - pooling_h + pad_top + pad_bottom) // stride_h + 1)
    out_width = simplify(
        (width - pooling_w + pad_left + pad_right) // stride_w + 1)
    dheight = hcl.reduce_axis(0, pooling_h)
    dwidth = hcl.reduce_axis(0, pooling_w)
    return hcl.compute(
        (batch, channel, out_height, out_width),
        lambda i, c, h, w: (sum(data[i, c, h *
                                     stride_h +
                                     dheight, w *
                                     stride_w +
                                     dwidth], axis=[dheight, dwidth]) /
                            (pooling_w *
                             pooling_h)),
        name=name,
        attrs=OrderedDict([
            ('out_img_w', out_width),
            ('out_img_h', out_height),
            ('in_num', channel),
            ('kernel_h', pooling[1]),
            ('kernel_w', pooling[0]),
            ('stride_h', stride[1]),
            ('stride_w', stride[0]),
            ('app_name', tvm.make.StringImm('avg_pool'))]))


def avg_pool2d_nhwc(
    data, pooling, stride=[
        1, 1], padding=[
            0, 0], name='avg_pool2d'):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    pooling_h, pooling_w = pooling
    stride_h, stride_w = stride
    batch, height, width, channel = data.shape
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(
        padding, (pooling_h, pooling_w))
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_bottom, pad_right, 0]
    data = pad(
        data,
        pad_before,
        pad_after,
        pad_value=tvm.const(
            0.0,
            data.dtype))
    out_height = simplify(
        (height - pooling_h + pad_top + pad_bottom) // stride_h + 1)
    out_width = simplify(
        (width - pooling_w + pad_left + pad_right) // stride_w + 1)
    dheight = hcl.reduce_axis(0, pooling_h)
    dwidth = hcl.reduce_axis(0, pooling_w)
    return hcl.compute(
        (batch, out_height, out_width, channel),
        lambda i, h, w, c: sum(data[i, h * stride_h + dheight, w * stride_w +
            dwidth, c], axis=[dheight, dwidth]) / (pooling_w * pooling_h),
            name=name,
            attrs=OrderedDict([
                ('out_img_w', out_width),
                ('out_img_h', out_height),
                ('in_num', channel),
                ('kernel_h', pooling[1]),
                ('kernel_w', pooling[0]),
                ('stride_h', stride[1]),
                ('stride_w', stride[0]),
                ('app_name', tvm.make.StringImm('avg_pool'))]))


def global_max_pool2d(data, layout='NCHW', name='global_max_pool2d'):
    stride = [1, 1]
    padding = [0, 0]
    if layout == 'NCHW':
        pooling = [data.shape[2], data.shape[3]]
        return max_pool2d_nchw(data, pooling, stride, padding, name)
    if layout == 'NHWC':
        pooling = [data.shape[1], data.shape[2]]
        return max_pool2d_nhwc(data, pooling, stride, padding, name)
    raise ValueError("not support this layout {} yet".format(layout))


def global_avg_pool2d(data, layout='NCHW', name='global_avg_pool2d'):
    stride = [1, 1]
    padding = [0, 0]
    if layout == 'NCHW':
        pooling = [data.shape[2], data.shape[3]]
        return avg_pool2d_nchw(data, pooling, stride, padding, name)
    if layout == 'NHWC':
        pooling = [data.shape[1], data.shape[2]]
        return avg_pool2d_nhwc(data, pooling, stride, padding, name)
    raise ValueError("not support this layout {} yet".format(layout))


def transpose(data, axes=[], name="transpose"):
    new_shape = []
    if len(axes) == 0:
        for i in range(len(data.shape)):
            axes.append(i)
    for i in range(len(axes)):
        new_axis = tvm_to_primitive(axes[i])
        if tvm_to_primitive(axes[i]) < 0:
            new_axis = len(data.shape) + tvm_to_primitive(axes[i])
            axes[i] = new_axis
        assert (
            new_axis >= 0 and new_axis < len(
                data.shape)), "axis={} is invalid for the {}-dimensional input tensor".format(
            new_axis, len(
                data.shape))
        for j in range(len(axes)):
            if (not i == j):
                assert(not new_axis == axes[j]), "repeated axis in transpose"
        new_shape.append(data.shape[new_axis])
    new_shape = tuple(new_shape)

    def _transpose(*indices):
        if len(axes) != 0:
            idx = [1] * len(axes)
            for i in range(len(axes)):
                idx[tvm_to_primitive(axes[i])] = indices[0][i]
        else:
            idx = indices[0]
        return idx
    return hcl.compute(new_shape, lambda *x: data[tuple(_transpose(x))], name=name,
        attrs=OrderedDict([('app_name',tvm.make.StringImm('transpose'))]))


def flatten(data, name="flatten"):
    ishape = data.shape
    dim = 1
    for i in range(1, len(ishape)):
        dim = dim * ishape[i]
    oshape = (ishape[0], dim)

    def unwrap(idx, shape):
        index = []
        for s in reversed(shape):
            index.append(idx % s)
            idx = idx / s
        return list(reversed(index))

    return hcl.compute(oshape, lambda i,j: data[tuple([i] + unwrap(j,ishape[1:]))],
        name=name,attrs=OrderedDict([('app_name',tvm.make.StringImm('flatten'))]))


def softmax(x, name="softmax", axis=0, frontend='keras'):
    shape = x.shape
    k = hcl.reduce_axis(0, shape[axis])
    new_shape = []
    for i in range(len(shape)):
        if i != axis:
            new_shape.append(shape[i])

    def _reduce_axis(axis, new_axis, keep_axis, *indices):
        indices = indices[0]
        new_ind = []
        put_axis = False
        for i in range(len(indices)):
            if i == axis and keep_axis:
                new_ind.append(new_axis)
                put_axis = True
                new_ind.append(indices[i])
            elif i != axis:
                new_ind.append(indices[i])
        if put_axis == False and keep_axis:
            new_ind.append(new_axis)
        return tuple(new_ind)
    max_elem = hcl.compute(
        tuple(new_shape), lambda *y: max(x[_reduce_axis(axis, k, True, y)], axis=[k]))
    k = hcl.reduce_axis(0, shape[axis])
    expsum = hcl.compute(
        tuple(new_shape), lambda *y: sum(tvm.exp(x[_reduce_axis(axis, k, True, y)] - max_elem[y]), axis=k))
    return hcl.compute(
        x.shape, lambda *y: tvm.exp(x[y] - max_elem[_reduce_axis(axis, k, False, y)]) / expsum[_reduce_axis(axis, k, False, y)], name)


def relu(data, name='relu'):
    return hcl.compute(data.shape,lambda *y: hcl.select(
        data[y] < 0, hcl.cast(data.dtype, 0), data[y]), name)


def leakyrelu(data, alpha=0.01):
    return hcl.compute(
        data.shape, lambda *y: hcl.select(data[y] < 0, alpha * data[y], data[y]))


def prelu(data, alpha, axis=1):
    def _axis_ind(axis, ind):
        ind = ind[0]
        new_ind = []
        for i in range(len(ind)):
            if i == axis:
                new_ind = ind[i]
        return tuple(new_ind)
    return hcl.compute(data.shape,lambda *x: hcl.select(
        data[x] < 0,
        hcl.cast(data.dtype, alpha[_axis_ind(axis,x)] * data[x]),
        data[x]))


def elu(data, alpha):
    return hcl.compute(data.shape,lambda *x: hcl.select(
        data[x] < 0, alpha * (hcl.exp(data[x])-1), data[x]))


def thresholdedrelu(data, theta):
    return hcl.compute(data.shape,lambda *x: hcl.select(
        data[x] > theta, data[x], hcl.cast(data.dtype, 0)))
