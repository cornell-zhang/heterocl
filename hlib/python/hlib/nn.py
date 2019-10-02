from collections import OrderedDict
import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np

dtype = hcl.Float()

sum = hcl.reducer(0, lambda x, y: x + y, dtype)
max = hcl.reducer(-1, lambda x, y: tvm.make.Max(x, y), dtype)
_all = hcl.reducer(True, lambda x, y: x & y, bool)

def tvm_to_prim(expr):
    if not isinstance(expr,int):
        return expr.value
    else:
        return expr

def simplify(expr):
    return tvm.ir_pass.Simplify(expr) if isinstance(
        expr, tvm.expr.Expr) else expr


def pad(data, pad_before, pad_after=None, pad_value=0.0, name="PadInput"):
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
            (data.shape[i] +
             tvm.const(
                pad_before[i] +
                pad_after[i]))) for i in range(n))
    pad_value = (pad_value if isinstance(pad_value, tvm.expr.Expr)
                 else tvm.const(pad_value, data.dtype))

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
    return hcl.compute(out_shape, _pad, name='pad')


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


def conv2d_transpose_nchw(
    data, kernel, strides=[
        1, 1], padding=[
            0, 0], out_dtype=None):
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
    dc = tvm.reduce_axis((0, in_c), name='dc')
    dh = tvm.reduce_axis((0, filter_h), name='dh')
    dw = tvm.reduce_axis((0, filter_w), name='dw')

    return hcl.compute(
        (batch, out_c, out_h, out_w),
        lambda b, c, h, w: hcl.sum(
            PaddedInput[b, dc, h + dh, w + dw].astype(out_dtype) *
            kernel[dc, c, filter_h - 1 - dh, filter_w - 1 - dw].astype(out_dtype),
            axis=[dc, dh, dw]), tag="conv2d_transpose_nchw")


def conv2d(
    Input,
    Filter,
    Bias=None,
    strides=[1, 1],
    padding=[0, 0],
    dilation=[1, 1],
    groups=1,
    channels=1,
    kernel_size=[1,1],
    data_layout='NCHW',
    kernel_layout='OIHW',
    out_layout='',
    name="conv2d",
    out_dtype=None):
    p = []
    s = []
    d = []
    for i in range(len(padding)):
        p.append(tvm_to_prim(padding[i]))
        s.append(tvm_to_prim(strides[i]))
        d.append(tvm_to_prim(dilation[i]))
    strides=s
    padding=p
    dilation=d
    if(out_dtype==None or out_dtype==''):
        out_dtype=Input.dtype
    if data_layout == 'NCHW':
        return conv2d_nchw(
            Input,
            Filter,
            strides,
            padding,
            dilation,
            name='conv2d',
            out_dtype=out_dtype)
    if data_layout == 'NHWC':
        return conv2d_nhwc(
            Input,
            Filter,
            strides,
            padding,
            dilation,
            name='conv2d',
            out_dtype=out_dtype)
    if data_layout == 'HWCN':
        return conv2d_hwcn(
            Input,
            Filter,
            strides,
            padding,
            dilation,
            name='conv2d',
            out_dtype=out_dtype)
    raise ValueError("not support this layout {} yet".format(layout))


def conv2d_nhwc(
    Input, Filter, strides=[
        1, 1], padding=[
            1, 1], dilation=[
                1, 1], out_dtype='float', name='conv2d'):
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
                 xx * stride_w + rx * dilation_w, rc].astype(out_dtype) *
            Filter[ry, rx, rc, ff].astype(out_dtype), axis=[ry, rx, rc],
            name=name))


def conv2d_nchw(
    Input, Filter, strides=[
        1, 1], padding=[
            0, 0], dilation=[
                1, 1], out_dtype=None, name='conv2d'):
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
    rc = hcl.reduce_axis(0, in_channel, name='rc')
    ry = hcl.reduce_axis(0, kernel_h, name='ry')
    rx = hcl.reduce_axis(0, kernel_w, name='rx')

    return hcl.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: hcl.sum(
            temp[nn, rc, yy * stride_h + ry * dilation_h,
                 xx * stride_w + rx * dilation_w].astype(out_dtype) *
            Filter[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx]), name=name)


def conv2d_hwcn(
    Input, Filter, strides=[
        1, 1], padding=[
            0, 0], dilation=[
                1, 1], out_dtype=None, name='conv2d'):
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
                 rc, nn].astype(out_dtype) *
            Filter[ry, rx, rc, ff].astype(out_dtype), axis=[ry, rx, rc]),
        name=name)


def conv2d_nchw_old(Input, Filter, name="conv2d", stride=[
                    1, 1], padding=[[0, 0], [0, 0]]):
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
    data_len = len(data.shape)
    bias_len = len(bias.shape)
    if(axis < 0):
        axis += data_len
    num_newaxis = data_len - axis - 1

    def _expand_dims(axis, new_axis, *indices):
        axes = []
        indices = indices[0]
        for i in range(axis):
            axes.append(indices[i])
        for i in range(len(indices) - new_axis):
            axes.append(indices[i + axis + new_axis])
        axes = tuple(axes)
        return axes
    if num_newaxis:
        b_add = hcl.compute(
            data.shape, lambda *x: data[x] + bias[_expand_dims(0, num_newaxis, x)], name=name)
    else:
        b_add = hcl.compute(data.shape,
                            lambda *x: data[x] + bias[_expand_dims(0,
                                                                   data_len - bias_len,
                                                                   x)],
                            name=name)
    return b_add


def expand_dims(data, axis, new_axis, name="expand_dims"):
    shape = []
    val_var = []
    ind_len = len(data.shape)
    for i in range(axis):
        shape.append(data.shape[i])
        val_var.append(1)
    for i in range(new_axis):
        shape.append(1)
        val_var.append(0)
    for i in range(ind_len - axis):
        shape.append(data.shape[i + axis])
        val_var.append(1)
    shape = tuple(shape)

    def _expand_ind(val_var, *indices):
        indices = indices[0]
        new_shape = []
        for i in range(len(val_var)):
            if val_var[i]:
                new_shape.append(indices[i])
        return tuple(new_shape)
    return hcl.compute(
        shape, lambda *x: data[_expand_ind(val_var, x)], name=name)


def squeeze(data, axis=None, name='squeeze'):
    if axis is None:
        axis = []
        for i in range(len(data.shape)):
            if data.shape[i] == 1:
                axis.append(i)
    l_orig = len(data.shape)
    new_shape = []
    for i in range(len(data.shape)):
        if i not in axis:
            new_shape.append(data.shape[i])

    def _ind(axis, l_orig, *indices):
        indices = indices[0]
        new_shape = []
        inx = 0
        for i in range(l_orig):
            if i not in axis:
                new_shape.append(indices[inx])
                inx = inx + 1
            else:
                new_shape.append(0)
        return tuple(new_shape)
    return hcl.compute(
        tuple(new_shape), lambda *x: data[_ind(axis, l_orig, x)], name=name)


def split(data, indices_or_sections, axis=0, name='split'):
    def split_inx(start, axis, *indices):
        indices = indices[0]
        new_ind = []
        for i in range(len(indices)):
            if i == axis:
                new_ind.append(start + indices[i])
            else:
                new_ind.append(indices[i])
        return tuple(new_ind)
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
                tuple(new_shape), lambda *x: data[split_inx(i * intval, axis, x)], name=name))
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
                new_shape[i]), lambda *x: data[split_inx(axis_width[i], axis, x)], name=name))
    return tuple(out)


def concatenate(*data_tup, axis=0, name='concatenate'):
    inx_start = [0]
    data_tup = list(data_tup[0])
    axis_len = 0
    for i in range(len(data_tup)):
        inx_start.append(inx_start[i] + (data_tup[i]).shape[axis])
        axis_len = axis_len + (data_tup[i]).shape[axis]
    new_shape = list(data_tup[0].shape)
    new_shape[axis] = axis_len
    C = hcl.placeholder(tuple(new_shape))

    def concat(data, offset, *indices):
        orig_inx = list(indices[0])
        inx = list(indices[0])
        inx[axis] = inx[axis] + offset
        orig_inx = tuple(orig_inx)
        inx = tuple(inx)
        C[inx] = data[orig_inx]
    for i in range(len(data_tup)):
        hcl.mutate(data_tup[i].shape,
                   lambda *x: concat(data_tup[i],
                                     inx_start[i],
                                     x),
                   name=name)
    return C

def red_mul(l):
    result = 1
    for item in l:
        result = result * item
    return result

def reshape(data, newshape, name='reshape'):
    res_shape = []
    cur_shape = list(data.shape)
    inx = 0
    inx_n1 = -1
    val_n1 = 1
    for _ in range(len(newshape)):
        new_inx = newshape[inx]
        assert(new_inx>-5), "inx has to be greater than -5"
        if(new_inx>0):
            res_shape.append(new_inx)
        elif(new_inx==0):
            res_shape.append(cur_shape[inx])
        elif(new_inx==-1):
            if(not inx_n1==-1):
                raise ValueError("no more than one -1 is allowed in newshape")
            inx_n1 = inx
        elif(new_inx==-2):
            res_shape.extend(cur_shape[inx:])
        elif(new_inx==-3):
            res_shape.append(cur_shape[inx]+cur_shape[inx+1])
            inx=inx+1
        elif(new_inx==-4):
            assert False, "not implemented yet"
        inx=inx+1
    if(not inx_n1 == -1):
        res_shape.insert(inx_n1,red_mul(cur_shape)//red_mul(res_shape))
    print(cur_shape,res_shape)
    assert(red_mul(cur_shape)==red_mul(res_shape)), "shape must contain same total product"
    cur_order = [1]
    res_order = [1]
    c_ord = len(cur_shape)-1
    for i in range(len(cur_shape)-1):
        cur_order.append(cur_shape[c_ord-i]*cur_order[i])
    r_ord = len(res_shape)-1
    for i in range(len(res_shape)-1):
        res_order.append(res_shape[r_ord-i]*res_order[i])
    def _reshape_inx(*indices):
        indices = indices[0]
        elm_inx = 0
        data_inx = []
        r_ord = len(res_order)-1
        for i in range(len(indices)):
            elm_inx = indices[i]*res_order[r_ord-i] + elm_inx
        c_ord = len(cur_order)-1
        for i in range(len(cur_order)):
            data_inx.append((elm_inx//(cur_order[i]))%cur_shape[i])
        print(data_inx)
        return tuple(data_inx)
    return hcl.compute(tuple(res_shape), lambda *x: data[_reshape_inx(x)], name=name)


def batch_norm(
        data,
        gamma,
        beta,
        moving_mean,
        moving_var,
        axis=1,
        epsilon=10**-7,
        center=1,
        scale=1):
    pass

# atm don't care about implementing this


def dropout(data, rate=0.5):
    data = hcl.compute(data.shape, lambda *x: data[x])
    mask = hcl.compute(data.shape, lambda *x: hcl.cast(dtype, 1))
    return data, mask
# def tanh(x, name="tanh"):
#    return hcl.compute(x.shape, lambda *args: hcl.tanh(x[args]), name,
# attrs=OrderedDict([('app_name', tvm.make.StringImm('tanh'))]))

# old version of max_pool


def max_pool(data, kernel, stride, padding=[
             [0, 0], [0, 0]], name="max_pool2d"):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    kernel_height, kernel_width = kernel
    stride_height, stride_width = stride
    batch, channel, height, width = data.shape
    [pad_top, pad_left], [pad_down, pad_right] = padding
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    if padding != [[0, 0], [0, 0]]:
        data = pad(
            data,
            pad_before,
            pad_after,
            pad_value=tvm.min_value(
                data.dtype))
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
        lambda i, c, h, w: max(data[i, c, h * stride_height + dheight, w * stride_width + dwidth], axis=[dheight, dwidth]),
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
    data, pool_size=[
        1, 1], strides=[
            1, 1], padding=[
                0, 0], layout='NCHW', name='max_pool2d'):
    pooling = []
    stride = []
    pad = []
    for i in range(len(pool_size)):
        pooling.append(tvm_to_prim(pool_size[i]))
        stride.append(tvm_to_prim(strides[i]))
        pad.append(tvm_to_prim(padding[i]))
    data = transpose(data,[0,3,1,2])
    if(len(pad)==4):
        pad = "SAME"
    if layout == 'NCHW':
        return transpose(max_pool2d_nchw(data, pooling, stride, pad, name),(0,2,3,1))
    if layout == 'NHWC':
        return transpose(max_pool2d_nhwc(data, pooling, stride, pad, name),(0,2,3,1))
    raise ValueError("not support this layout {} yet".format(layout))


def max_pool2d_nchw(data, pooling, stride, padding, name='max_pool2d'):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    max = hcl.reducer(tvm.min_value(data.dtype), lambda x, y: tvm.make.Max(x, y), data.dtype)
    pooling_h, pooling_w = pooling
    stride_h, stride_w = stride
    batch, channel, height, width = data.shape
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(
        padding, (pooling_h, pooling_w))
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_bottom, pad_right]
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
        (batch, channel, out_height, out_width),
        lambda i, c, h, w: max(data[i, c, h * stride_h + dheight, w * stride_w + dwidth], axis=[dheight, dwidth]),
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
    data, pooling, stride=[
        1, 1], padding=[
            0, 0], name='max_pool2d'):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    max = hcl.reducer(tvm.min_value(data.dtype), lambda x, y: tvm.make.Max(x, y), data.dtype)
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
        lambda i, h, w, c: max(data[i, h * stride_h + dheight, w * stride_w + dwidth, c], axis=[dheight, dwidth]),
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
    data, pool_size=[
        1, 1], strides=[
            1, 1], padding=[
                0, 0], layout='NCHW', name='avg_pool2d'):
    pooling = []
    stride = []
    pad = []
    for i in range(len(pool_size)):
        pooling.append(tvm_to_prim(pool_size[i]))
        stride.append(tvm_to_prim(strides[i]))
        pad.append(tvm_to_prim(padding[i]))
    data = transpose(data,[0,3,1,2])
    if layout == 'NCHW':
        return transpose(avg_pool2d_nchw(data, pooling, stride, pad, name),(0,2,3,1))
    if layout == 'NHWC':
        return transpose(avg_pool2d_nhwc(data, pooling, stride, pad, name),(0,2,3,1))
    raise ValueError("not support this layout {} yet".format(layout))


def avg_pool2d_nchw(data, pooling, stride, padding, name='avg_pool2d'):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    pooling_h, pooling_w = pooling
    stride_h, stride_w = stride
    batch, channel, height, width = data.shape
    if(len(padding)==4):
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
    print(pooling_h,pooling_w)
    print(stride_h,stride_w)
    return hcl.compute(
        (batch, channel, out_height, out_width),
        lambda i, c, h, w: (sum(data[i, c, h * stride_h + dheight, w * stride_w + dwidth], axis=[dheight, dwidth]) / (pooling_w * pooling_h)),
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
        lambda i, h, w, c: sum(data[i, h * stride_h + dheight, w * stride_w + dwidth, c], axis=[dheight, dwidth]) / (pooling_w * pooling_h),
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
    if(len(axes) == 0):
        for i in range(len(data.shape)):
            axes.append(i)
    for i in range(len(axes)):
        axis = axes[i]
        new_axis = axis
        if(axis < 0):
            new_axis = len(data.shape) + axis
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
        if(len(axes) != 0):
            idx = [1] * len(axes)
            for i in range(len(axes)):
                idx[axes[i]] = indices[0][i]
        else:
            idx = indices[0]
        return idx
    return hcl.compute(new_shape,
                       lambda *x: data[tuple(_transpose(x))],
                       name=name,
                       attrs=OrderedDict([('app_name',
                                           tvm.make.StringImm('transpose'))]))


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

    return hcl.compute(oshape,
                       lambda i,
                       j: data[tuple([i] + unwrap(j,
                                                  ishape[1:]))],
                       name=name,
                       attrs=OrderedDict([('app_name',
                                           tvm.make.StringImm('flatten'))]))


def softmax(x, name="softmax", axis=0):
    assert len(x.shape) == 2, "only support 2-dim softmax"
    m, n = x.shape
    k = hcl.reduce_axis(0, n)
    max_elem = hcl.compute((m, ), lambda i: max(x[i, k], axis=k))
    k = hcl.reduce_axis(0, n)
    expsum = hcl.compute(
        (m, ), lambda i: sum(tvm.exp(x[i, k] - max_elem[i]), axis=k))
    return hcl.compute(
        x.shape, lambda i, j: tvm.exp(x[i, j] - max_elem[i]) / expsum[i], name)


def relu(data, name='relu'):
    return hcl.compute(data.shape,
                       lambda *y: hcl.select(data[y] < 0,
                                             hcl.cast(data.dtype,
                                                      0),
                                             data[y]),
                       name)


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
    return hcl.compute(data.shape,
                       lambda *x: hcl.select(data[x] < 0,
                                             hcl.cast(data.dtype,
                                                      alpha[_axis_ind(axis,
                                                                      x)] * data[x]),
                                             data[x]))


def elu(data, alpha):
    return hcl.compute(data.shape,
                       lambda *x: hcl.select(data[x] < 0,
                                             alpha * (hcl.exp(data[x]) - 1),
                                             data[x]))


def thresholdedrelu(data, theta):
    return hcl.compute(data.shape,
                       lambda *x: hcl.select(data[x] > theta,
                                             data[x],
                                             hcl.cast(data.dtype,
                                                      0)))
