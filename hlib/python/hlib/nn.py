from collections import OrderedDict
import heterocl as hcl
import heterocl.tvm as tvm
from numbers import Integral

def equal_const_int(expr, value):
    if isinstance(expr, Integral):
        return expr == value
    if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        expr = tvm.ir_pass.Simplify(expr)
    if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
        return False
    return expr.value == value

dtype = hcl.Float()

sum = hcl.reducer(0, lambda x, y: x + y, dtype)
max = hcl.reducer(-1, lambda x, y: tvm.make.Max(x, y), dtype)

def simplify(expr):
    return tvm.ir_pass.Simplify(expr) if isinstance(expr, tvm.expr.Expr) else expr

def pad(data, pad_before, pad_after=None, pad_value=0.0, name="pad"):
    n = len(data.shape)
    pad_after = pad_after if pad_after else pad_before
    out_shape = tuple(
        tvm.ir_pass.Simplify(
            (data.shape[i] + tvm.const(pad_before[i]) + tvm.const(pad_after[i]))) for i in range(n))
    def _pad(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if equal_const_int(pad_before[i], 0) and equal_const_int(pad_after[i], 0):
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

def conv2d_nchw_imp(Input, Filter, Output, stride=[1,1], padding=[[0,0],[0,0]]):
    with hcl.for_(0,Output.shape[0], name="n") as n:
      with hcl.for_(0,Output.shape[1], name="c") as c:
        with hcl.for_(0,Output.shape[2], name="h") as h:
          with hcl.for_(0,Output.shape[3], name="w") as w:
            partial = hcl.scalar(0)
            with hcl.for_(0,Filter.shape[-2], name="x") as x:
              with hcl.for_(0,Filter.shape[-1], name="y") as y:
                partial.v += Input[n][c][h+x][w+y] * Filter[0][0][x][y] 
            Output[n,c,h,w] = partial

def conv2d_nchw(Input, Filter, name="conv2d", stride=[1,1], 
                padding=[[0,0],[0,0]], activation="relu"):
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
        Input = pad(Input, pad_before, pad_after, name=name+"_pad")
    rc = hcl.reduce_axis(0, in_channel)
    ry = hcl.reduce_axis(0, kernel_h)
    rx = hcl.reduce_axis(0, kernel_w)

    actv = lambda x: x
    if activation == "relu":
      actv = lambda x: hcl.select(x > 0, x, 0)
    if activation == "tanh":
      actv = lambda x: tvm.tanh(x)

    return hcl.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: actv(sum(
            Input[nn, rc, yy * stride_h + ry, xx * stride_w + rx] *
            Filter[ff, rc, ry, rx],
            axis=[rc, ry, rx])),
        name=name,
        attrs=OrderedDict([
            ('p', kernel_h),
            ('q', kernel_w),
            ('in_num', in_channel),
            ('out_num', out_channel),
            ('stride_h', stride[1]),
            ('stride_w', stride[0]),
            ('out_img_w', out_width),
            ('out_img_h', out_height),
            ('cin_dtype', tvm.make.StringImm(Input.dtype)),
            ('filter_dtype', tvm.make.StringImm(Filter.dtype)),
            ('app_name', tvm.make.StringImm('cnn'))]))

def dense(data, weight, bias=None, name="dense", activation="relu"):
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1

    actv = lambda x: x
    if activation == "relu":
      actv = lambda x: hcl.select(x > 0, x, 0)
    if activation == "tanh":
      actv = lambda x: tvm.tanh(x)

    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = hcl.reduce_axis(0, in_dim)
    attrs=OrderedDict([
        ('k', in_dim),
        ('j', out_dim),
        ('i', batch),
        ('app_name', tvm.make.StringImm('mm'))])
    if bias is None:
        return hcl.compute((batch, out_dim), 
            lambda i, j: actv(sum(
                data[i, k] * weight[j, k], axis=k)), name, attrs=attrs)
    else: # bias per dimension
        return hcl.compute((batch, out_dim), 
            lambda i, j: actv(sum(
                data[i, k] * weight[j, k] + bias[j], axis=k)), name, attrs=attrs)

def tanh(x, name="tanh"):
    return hcl.compute(x.shape, lambda *args: tvm.tanh(x[args]), name,
                       attrs=OrderedDict([('app_name', tvm.make.StringImm('tanh'))]))

def relu(x, name="relu"):
    return hcl.compute(x.shape, lambda *args: hcl.select(x[args] > 0.0, x[args], 0.0), name,
                       attrs=OrderedDict([('app_name', tvm.make.StringImm('relu'))]))

def tensoradd(x, y, name='tensoradd'):
    assert x.shape == y.shape, 'x, y must be of same shape'
    return hcl.compute(x.shape, lambda *args: x[args] + y[args], name,
                       attrs=OrderedDict([('app_name', tvm.make.StringImm('tensoradd'))]))

def max_pool(data, kernel, stride, padding=[[0,0],[0,0]], name="max_pool"):
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

# default: inter-channel local response norm
def local_resp_norm(x, k, alpha, beta, window=2, name="lrn", inter=True):
    assert window % 2 == 0, "only support 2's multiple window size"
    r = hcl.reduce_axis(-window/2, window/2, name="rdx")
    if inter: # inter channel lrn
        size = x.shape[1]
        return hcl.compute(x.shape,
            lambda n, c, h, w:
                x[n, c, h, w] / (k + alpha * sum(
                    hcl.power(hcl.select(hcl.and_(c+r>=0, c+r<=size-1), x[n, c+r, h, w], 0), 2), axis=r)),
            name, attrs=OrderedDict([('app_name', tvm.make.StringImm('local_resp_norm'))]))

def batch_norm(x, beta, gamma, mean, var, name="batch_norm"):
    assert len(x.shape) == 4, 'batch norm for 4d tensor'
    return hcl.compute(x.shape, 
        lambda n, c, h, w: 
            (x[n, c, h, w] - mean[c]) / hcl.sqrt(var[c]) * gamma[c] + beta[c], name,
        attrs=OrderedDict([('app_name', tvm.make.StringImm('batch_norm'))]))

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

    return hcl.compute(oshape, lambda i, j: data[tuple([i] + unwrap(j, ishape[1:]))], name,
                       attrs=OrderedDict([('app_name', tvm.make.StringImm('flatten'))]))

def softmax(out, x):
    assert len(x.shape) == 2, "only support 2-dim softmax"
    m, n = x.shape
    k = hcl.reduce_axis(0, n)
    max_elem = hcl.compute((m, ), lambda i: max(x[i, k], axis=k))
    k = hcl.reduce_axis(0, n)
    expsum = hcl.compute(
        (m, ), lambda i: sum(tvm.exp(x[i, k] - max_elem[i]), axis=k))
    return hcl.update(
        out, lambda i, j: tvm.exp(x[i, j] - max_elem[i]) / expsum[i])

