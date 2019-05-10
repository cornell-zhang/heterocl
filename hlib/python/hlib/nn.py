from collections import OrderedDict
import heterocl as hcl
import heterocl.tvm as tvm

dtype = hcl.Float()

sum = hcl.reducer(0, lambda x, y: x + y, dtype)
max = hcl.reducer(-1, lambda x, y: tvm.make.Max(x, y), dtype)

def simplify(expr):
    return tvm.ir_pass.Simplify(expr) if isinstance(expr, tvm.expr.Expr) else expr

#def pad(data, pad_before, pad_after=None, pad_value=0.0):
#    n = len(data.shape)
#    pad_after = pad_after if pad_after else pad_before
#    out_shape = tuple(
#        tvm.ir_pass.Simplify(
#            (data.shape[i] + tvm.const(pad_before[i]) + tvm.const(pad_after[i]))) for i in range(n))
#    def _pad(*indices):
#        not_zero = []
#        index_tuple = []
#        for i in range(n):
#            if equal_const_int(pad_before[i], 0) and equal_const_int(pad_after[i], 0):
#                index_tuple.append(indices[i])
#            else:
#                index_tuple.append(indices[i] - pad_before[i])
#                not_zero.append(indices[i] >= pad_before[i])
#                not_zero.append(indices[i] < data.shape[i] + pad_before[i])
#        if not_zero:
#            not_zero = tvm.all(*not_zero)
#            return tvm.select(not_zero, data[tuple(index_tuple)], pad_value)
#        return data[tuple(index_tuple)]
#    return hcl.compute(out_shape, _pad, name='pad')

def pad(data, pad_before, pad_after=None, pad_value=0.0, name="PadInput"):
    n = len(data.shape)
    pad_after = pad_after if pad_after else pad_before
    if len(pad_before) != n:
        raise ValueError("Input dimension and pad_before dismatch : %d vs %d" % (
            n, len(pad_before)))
    if len(pad_after) != n:
        raise ValueError("Input dimension and pad_after dismatch : %d vs %d" % (
            n, len(pad_after)))
    out_shape = tuple(
        tvm.ir_pass.Simplify(
            (data.shape[i] +tvm.const(pad_before[i] + pad_after[i]))) for i in range(n))
    pad_value = (pad_value if isinstance(pad_value, tvm.expr.Expr)
                else tvm.const(pad_value, data.dtype))
    def _pad(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if pad_before[i] == 0 and pad_after[i]== 0:
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
    pad_top  = (pad_h + 1 // 2)
    pad_left = (pad_w + 1 // 2)
    return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left

def conv2d(Input, Filter, Bias=None, stride=[1,1], padding=[0,0], dilation=[1,1], layout='NCHW', name="conv2d", out_dtype=None ):
    if layout == 'NCHW':
        return _conv2d_nchw(Input, Filter, Bias, stride, padding, dilation, name='conv2d', out_dtype=out_dtype )
    if layout == 'NHWC':
        return _conv2d_nhwc(Input, Filter, Bias, stride, padding, dilation, name='conv2d', out_dtype=out_dtype )
    raise ValueError("not support this layout {} yet".format(layout))

def _conv2d_nchw(Input, Filter, Bias=None, stride=[1,1], padding=[0,0], dilation=[1,1], name='conv2d', out_dtype=None ):
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w  = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w  = dilation

    batch, in_channel, in_height, in_width  = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    #compute output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
)
    pad_before = [0,0,pad_top, pad_left]
    pad_after = [0,0,pad_down, pad_right]
    temp = pad(Input, pad_before, pad_after, name="pad_temp")
    rc = hcl.reduce_axis(0, in_channel)
    ry = hcl.reduce_axis(0, kernel_h)
    rx = hcl.reduce_axis(0, kernel_w)

    if not Bias==None:
      return hcl.compute(
          (batch, out_channel, out_height, out_width),
          lambda nn, ff, yy, xx: hcl.sum(
              temp[nn, rc, yy * stride_h + ry * dilation_h,
                  xx * stride_w + rx * dilation_w].astype(out_dtype) * 
              Filter[ff, rc, ry, rx].astype(out_dtype) + Bias[ff].astype(out_dtype),
              axis=[rc, ry, rx]), name=name,
          attrs=OrderedDict([
              ('p', kernel_h),
              ('q', kernel_w),
              ('in_num', in_channel),
              ('out_num', out_channel),
              ('out_img_w', out_width),
              ('out_img_h', out_height),
              ('cin_dtype', tvm.make.StringImm(Input.dtype)),
              ('filter_dtype', tvm.make.StringImm(Filter.dtype)),
	      ('bias_dtype', tvm.make.StringImm(Bias.dtype)),
              ('app_name', tvm.make.StringImm('cnn'))]))
    return hcl.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: hcl.sum(
            temp[nn, rc, yy * stride_h + ry * dilation_h,
                xx * stride_w + rx * dilation_w].astype(out_dtype) * 
            Filter[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx]), name=name,
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

def _conv2d_nhwc(Input, Filter, Bias=None, stride=[1,1], padding=[0,0], dilation=[1,1], name='conv2d', out_dtype=None ):
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w  = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w  = dilation

    batch, in_height, in_width, in_channel  = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    #compute output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
)
    pad_before = [0,pad_top, pad_left,0]
    pad_after = [0,pad_down, pad_right,0]
    temp = pad(Input, pad_before, pad_after, name="pad_temp")
    rc = hcl.reduce_axis(0, in_channel)
    ry = hcl.reduce_axis(0, kernel_h)
    rx = hcl.reduce_axis(0, kernel_w)

    if not Bias==None:
      return hcl.compute(
          (batch, out_height, out_width, out_channel),
          lambda nn, yy, xx, ff: hcl.sum(
              temp[nn, yy * stride_h + ry * dilation_h,
                  xx * stride_w + rx * dilation_w, rc].astype(out_dtype) * 
              Filter[ff, rc, ry, rx].astype(out_dtype) + Bias[ff].astype(out_dtype),
              axis=[ry, rx, rc]), name=name,
          attrs=OrderedDict([
              ('p', kernel_h),
              ('q', kernel_w),
              ('in_num', in_channel),
              ('out_num', out_channel),
              ('out_img_w', out_width),
              ('out_img_h', out_height),
              ('cin_dtype', tvm.make.StringImm(Input.dtype)),
              ('filter_dtype', tvm.make.StringImm(Filter.dtype)),
	      ('bias_dtype', tvm.make.StringImm(Bias.dtype)),
              ('app_name', tvm.make.StringImm('cnn'))]))
    return hcl.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, yy, xx, ff: hcl.sum(
            temp[nn, yy * stride_h + ry * dilation_h,
                xx * stride_w + rx * dilation_w, rc].astype(out_dtype) * 
            Filter[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx]), name=name,
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




def conv2d_nchw(Input, Filter, name="conv2d", stride=[1,1], padding=[[0,0],[0,0]]):
    out_dtype = Input.dtype
    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape
    stride_h, stride_w = stride
    [pad_top, pad_left], [pad_down, pad_right] = padding
    # compute the output shape
    out_channel = num_filter
    out_height = simplify((in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    if padding != [[0,0],[0,0]]:
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

def dense(data, weight, bias=None, name="dense"):
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = hcl.reduce_axis(0, in_dim)
    attrs=OrderedDict([
        ('k', in_dim),
        ('j', out_dim),
        ('i', batch),
        ('app_name', tvm.make.StringImm('mm'))])
    matmul = hcl.compute((batch, out_dim), lambda i, j: sum(data[i, k] * weight[j, k], axis=k), name, attrs=attrs)
    if bias is not None:
        matmul = hcl.compute(
                (batch, out_dim),
                lambda i, j: matmul[i, j] + bias[j],
                name=name,
                attrs=attrs)
    return matmul

def tanh(x, name="tanh"):
    return hcl.compute(x.shape, lambda *args: hcl.tanh(x[args]), name,
                       attrs=OrderedDict([('app_name', tvm.make.StringImm('tanh'))]))

#old version of max_pool
def max_pool(data, kernel, stride, padding=[[0,0],[0,0]],  name="max_pool2d"):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    kernel_height, kernel_width = kernel
    stride_height, stride_width = stride
    batch, channel, height, width = data.shape
    [pad_top, pad_left], [pad_down, pad_right] = padding
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    if padding != [[0,0],[0,0]]:
        data = pad(data, pad_before, pad_after, pad_value=tvm.min_value("float32"))
    out_height = simplify((height - kernel_height + pad_top + pad_down) // stride_height + 1)
    out_width = simplify((width - kernel_width + pad_left + pad_right) // stride_width + 1)
    dheight = hcl.reduce_axis(0, kernel_height)
    dwidth = hcl.reduce_axis(0, kernel_width)

    return hcl.compute(
        (batch, channel, out_height, out_width),
        lambda i, c, h, w: max(data[i, c, h*stride_height+dheight, w*stride_width+dwidth], axis=[dheight, dwidth]),
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

def max_pool2d(data, pooling, stride=[1,1], padding=[0,0], layout='NCHW',name='max_pool2d'):
    if layout=='NCHW':
        return max_pool2d_nchw(data,pooling,stride,padding,name)
    if layout=='NHWC':
        return max_pool2d_nhwc(data,pooling,stride,padding,name)
    raise ValueError("not support this layout {} yet".format(layout))

def max_pool2d_nchw(data, pooling, stride, padding, name='max_pool2d' ):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride)     == 2, "only support 2-dim stride"
    pooling_h,pooling_w  = pooling
    stride_h, stride_w   = stride
    batch, channel, height, width = data.shape
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(
      padding, (pooling_h,pooling_w))
    pad_before = [0, 0, pad_top, pad_left]
    pad_after  = [0, 0, pad_bottom,pad_right]
    if padding != [0,0]:
      data = pad(data, pad_before, pad_after, pad_value=tvm.min_value("float32"))
    out_height = simplify(
      (height - pooling_h + pad_top + pad_bottom) // stride_h + 1)
    out_width  = simplify(
      (width  - pooling_w + pad_left + pad_right) // stride_w + 1)
    dheight = hcl.reduce_axis(0, pooling_h)
    dwidth  = hcl.reduce_axis(0, pooling_w)
    return hcl.compute(
        (batch, channel, out_height, out_width),
        lambda i, c, h, w: max(data[i, c, h*stride_h+dheight, w*stride_w+dwidth], axis=[dheight, dwidth]),
        name=name,
        attrs=OrderedDict([
            ('out_img_w', out_width ),
            ('out_img_h', out_height),
            ('in_num',    channel   ),
            ('kernel_h', pooling[1] ),
            ('kernel_w', pooling[0] ),
            ('stride_h', stride[1]  ),
            ('stride_w', stride[0]  ),
            ('app_name', tvm.make.StringImm('max_pool'))]))

def max_pool2d_nhwc(data, pooling, stride=[1,1], padding=[0,0], name='max_pool2d' ):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride)     == 2, "only support 2-dim stride"
    pooling_h,pooling_w  = pooling
    stride_h, stride_w   = stride
    batch, height, width, channel = data.shape
    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(
      padding, (pooling_h,pooling_w))
    pad_before = [0, pad_top, pad_left, 0]
    pad_after  = [0, pad_bottom,pad_right, 0]
    if padding != [0,0]:
      data = pad(data, pad_before, pad_after, pad_value=tvm.min_value("float32"))
    out_height = simplify(
      (height - pooling_h + pad_top + pad_bottom) // stride_h + 1)
    out_width  = simplify(
      (width  - pooling_w + pad_left + pad_right) // stride_w + 1)
    dheight = hcl.reduce_axis(0, pooling_h)
    dwidth  = hcl.reduce_axis(0, pooling_w)
    return hcl.compute(
        (batch, out_height, out_width, channel),
        lambda i, h, w, c: max(data[i, h*stride_h+dheight, w*stride_w+dwidth, c], axis=[dheight, dwidth]),
        name=name,
        attrs=OrderedDict([
            ('out_img_w', out_width ),
            ('out_img_h', out_height),
            ('in_num',    channel   ),
            ('kernel_h', pooling[1] ),
            ('kernel_w', pooling[0] ),
            ('stride_h', stride[1]  ),
            ('stride_w', stride[0]  ),
            ('app_name', tvm.make.StringImm('max_pool'))]))

def transpose(data,axes,name="transpose"):
    new_shape = []
    if(len(axes) == 0):
      for i in data.shape:
        axes.append(i)
    for i in range(len(axes)):
      axis = axes[i]
      new_axis = axis
      if (axis < 0):
        new_axis = len(data.shape) + axis
        axes[i] = new_axis
      assert (new_axis >= 0 and new_axis < len(data.shape)), "axis={} is invalid for the {}-dimensional input tensor".format(new_axis,len(data.shape))
      for j in range(len(axes)):
        if ( not i == j ):
          assert(not new_axis == axes[j]), "repeated axis in transpose"
      new_shape.append(data.shape[new_axis])
    new_shape = tuple(new_shape)
    def _transpose(*indices):
      idx = [1]*len(axes)
      for i in range(len(axes)):
        idx[axes[i]] = indices[0][i]
      return idx
    return hcl.compute(new_shape, lambda *x: data[tuple(_transpose(x))],
                       name=name,
                       attrs=OrderedDict([('app_name', tvm.make.StringImm('transpose'))]))

def flatten(data,name="flatten"):
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

    return hcl.compute(oshape, lambda i, j: data[tuple([i] + unwrap(j, ishape[1:]))], name=name,
                       attrs=OrderedDict([('app_name', tvm.make.StringImm('flatten'))]))

def softmax(x,name="softmax"):
    assert len(x.shape) == 2, "only support 2-dim softmax"
    m, n = x.shape
    k = hcl.reduce_axis(0, n)
    max_elem = hcl.compute((m, ), lambda i: max(x[i, k], axis=k))
    k = hcl.reduce_axis(0, n)
    expsum = hcl.compute(
        (m, ), lambda i: sum(tvm.exp(x[i, k] - max_elem[i]), axis=k))
    return hcl.compute(
        x.shape, lambda i, j: tvm.exp(x[i, j] - max_elem[i]) / expsum[i],name)

def relu(x,name ='relu'):
    return hcl.compute(
        x.shape, lambda *y: hcl.select(x[y] < 0,hcl.cast(x.dtype,0),x[y]),name)

def leakyrelu(out, x, alpha=0.01):
    assert len(x.shape) == 2, "only support 2-dim LeakyReLU"
    m, n = x.shape
    k = hcl.reduce_axis(0,n)
    return hcl.update(
	out, lambda i,j: hcl.select(x[i,j] < 0,alpha*x[i,j],x[i,j]))

def prelu(out, x, alpha):
    assert len(x.shape) == 2, "only support 2-dim PReLU"
    m, n = x.shape
    k = hcl.reduce_axis(0,n)
    return hcl.update(
        out, lambda i,j: hcl.select(x[i,j] < 0,hcl.cast(x.dtype,alpha[j]*x[i,j]),x[i,j]))

def elu(out, x, alpha):
    assert len(x.shape) == 2, "only support 2-dim ELU"
    m, n = x.shape
    k = hcl.reduce_axis(0,n)
    return hcl.update(out, lambda i,j: hcl.select(x[i,j] < 0,alpha*(hcl.exp(x[i,j])-1),x[i,j]))

def thresholdedrelu(out, x, theta):
    assert len(x.shape) == 2, "only support 2-dim ThresholdedReLU"
    m, n = x.shape
    k = hcl.reduce_axis(0,n)
    return hcl.update(out, lambda i,j: hcl.select(x[i,j]>theta,x[i,j],hcl.cast(x.dtype,0)))
