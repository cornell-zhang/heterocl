import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np
from ..utils import *
from .op import *
from .nn import pad, get_pad_tuple, simplify

dtype = hcl.Float()
qtype_bit = hcl.UInt(1)

def if_mac(y, x, in_h, in_w, pad_top, pad_left, pad_down, pad_right):
    return tvm.all(x >= pad_left, x < in_w - pad_right, y >= pad_top, y < in_h - pad_down)

def pad_(data, padding=[1,1], name="pad", dtype=None):
    assert len(data.shape) == 4, "Only support 4D padding"
    if dtype == None:
        dtype = data.dtype
    batch, channel, in_height, in_width = data.shape
    out_height, out_width = in_height + 2 * padding[0], in_width + 2 * padding[1]
    return hcl.compute((batch,channel,out_height,out_width),
                lambda ii, cc, hh, ww: hcl.select(if_mac(hh,ww,out_height,out_width,padding[0],padding[1],padding[0],padding[1]),
                data[ii, cc, hh-padding[0], ww-padding[1]],0),
                dtype=dtype,name=name)

def pad_nhwc(data, padding=[1,1], name="pad", dtype=None):
    assert len(data.shape) == 4, "Only support 4D padding"
    if dtype == None:
        dtype = data.dtype
    batch, in_height, in_width, channel = data.shape
    out_height, out_width = in_height + 2 * padding[0], in_width + 2 * padding[1]
    return hcl.compute((batch,out_height,out_width,channel),
                lambda ii, hh, ww, cc: hcl.select(if_mac(hh,ww,out_height,out_width,padding[0],padding[1],padding[0],padding[1]),
                data[ii, hh-padding[0], ww-padding[1], cc],0),
                dtype=dtype,name=name)

def flatten(data, name="flatten"):
    ishape = data.shape
    dim = 1
    for i in range(1, len(ishape)):
        dim = dim * ishape[i]
    oshape = (ishape[0], dim)

    def unwrap(idx, shape): # channel first
        index = [idx % shape[0], idx / (shape[0]*shape[1]), (idx / shape[0]) % shape[1]]
        return index

    return hcl.compute(oshape, lambda i,j: data[tuple([i] + unwrap(j,ishape[1:]))],
        name=name,
        dtype=data.dtype)

def packed_flatten(data, name="packed_flatten"):
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
        name=name)

def packed_flatten_nhwc(data, name="packed_flatten"):
    batch, in_height, in_width, channel = data.shape
    out_shape = (batch, in_height * in_width * channel)
    return hcl.compute(out_shape, lambda i, j: data[i, j / (in_width * channel) % in_height, j / channel % in_width, j % channel],
        name=name)

def dense(data, weight, bias=None, use_relu=False, name="binary_dense"):
    assert len(
        data.shape) == 2 and len(
        weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = hcl.reduce_axis(0, in_dim)
    var_w = np.sqrt(2. / in_dim) # predefined constant
    # var_w = 1
    if bias is None:
        matmul = hcl.compute((batch, out_dim), lambda i, j: sum(
            tvm.all(data[i, k] == weight[j, k]), axis=k)
            * 2 - in_dim,
            name=name+"_matmul") # Data type needs to be specified!
    else:
        matmul = hcl.compute((batch, out_dim), lambda i, j: (hcl.sum(
            tvm.all(data[i, k] == weight[j, k]), axis=k, dtype=bias.dtype, name=name+"_sum")
            * 2 - in_dim) * var_w + bias[j],
            name=(name+"_matmul" if use_relu else name),
            dtype=bias.dtype)
    if use_relu:
        matmul = hcl.compute(
            (batch, out_dim),
            lambda i, j: hcl.select(matmul[i, j] > 0, 1, 0),
            name=name,
            dtype=qtype_bit
        )
    return matmul

def _popcount(num,bitwidth,name="popcnt"):
    out = hcl.scalar(0, name=name)
    with hcl.for_(0, bitwidth) as i:
        # Bit selection operation
        out.v += num[i]
    return out.v

def packed_dense(data, weight, bias=None, use_relu=False, name="packed_binary_dense", dtype=None):
    assert len(
        data.shape) == 2 and len(
        weight.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    assert "int" in data.dtype, "data type should be int or unsigned int"
    bitwidth = int(data.dtype.split("int")[-1])
    batch, in_dim = data.shape # in_dim has been packed
    out_dim, _ = weight.shape # only packed axis 1
    rk = hcl.reduce_axis(0, in_dim, name=name+"_rk")
    var_w = np.sqrt(2. / in_dim) # predefined constant
    # var_w = 1
    rb = hcl.reduce_axis(0, bitwidth, name=name+"_rb")
    if bias is not None:
        matmul = hcl.compute((batch, out_dim), lambda i, j:
                sum((data[i, rk] ^ weight[j, rk])[rb], # popcount
                axis=[rk, rb],name=name+"_popcnt",dtype=data.dtype),
                name=name+"_matmul",dtype=data.dtype)
    if not use_relu:
        matmul = hcl.compute((batch, out_dim), lambda i, j:
                (in_dim * bitwidth - (matmul[i, j] << 1)) * var_w + bias[j],
                name=name,
                dtype=bias.dtype if dtype==None else dtype)
    else:
        def genpack(i, j):
            out = hcl.scalar(0, name=name+"_pack", dtype=data.dtype)
            with hcl.for_(0, bitwidth) as k:
                out[0][(k+1) : k] = hcl.select(((in_dim * bitwidth - (matmul[i, j*bitwidth+k] << 1)) * var_w + bias[j*bitwidth+k]) > 0, 1, 0)
            return out[0]
        matmul = hcl.compute(
            (batch, out_dim // bitwidth),
            genpack,
            name=name,
            dtype=data.dtype if dtype==None else dtype
        )
    return matmul

def conv2d_nchw(
        Input,
        Filter,
        strides=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        out_dtype=None,
        name='binary_conv2d'):
    if out_dtype is None or out_dtype == '':
        out_dtype = hcl.Int()
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
    temp = pad(Input, pad_before, pad_after, name=name+"_pad")
    pad_in_height = in_height + pad_top + pad_down
    pad_in_width = in_width + pad_left + pad_right
    rc = hcl.reduce_axis(0, channel, name='rc')
    ry = hcl.reduce_axis(0, kernel_h, name='ry')
    rx = hcl.reduce_axis(0, kernel_w, name='rx')
    if channel > 1:
        out = hcl.compute(
            (batch, out_channel, out_height, out_width),
            lambda nn, ff, yy, xx: hcl.sum(
                hcl.select(
                    if_mac(yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, pad_in_height, pad_in_width, pad_top, pad_left, pad_down, pad_right), # neglect padding pixels in mac
                    ((1 - temp[nn, rc, yy * stride_h + ry * dilation_h,
                                xx * stride_w + rx * dilation_w] ^
                            Filter[ff, rc, ry, rx])
                    << 1) - 1, # xnor
                    0),
                axis=[rc, ry, rx], dtype=out_dtype, name=name+"_sum"),
                name=name,
                dtype=out_dtype)
    else: # TODO: otherwise, reuse_at may cause bug
        out = hcl.compute(
            (batch, out_channel, out_height, out_width),
            lambda nn, ff, yy, xx: hcl.sum(
                hcl.select(
                    if_mac(yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, pad_in_height, pad_in_width, pad_top, pad_left, pad_down, pad_right), # neglect padding pixels in mac
                    ((1 - temp[nn, 0, yy * stride_h + ry * dilation_h,
                                xx * stride_w + rx * dilation_w] ^
                            Filter[ff, 0, ry, rx])
                    << 1) - 1, # xnor
                    0),
                axis=[ry, rx], dtype=out_dtype, name=name+"_sum"),
                name=name,
                dtype=out_dtype)
    return out

def packed_conv2d_nchw(
        Input,
        Filter,
        strides=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        out_dtype=None,
        threshold=None,
        bitwidth=None,
        mac=True,
        name='packed_binary_conv2d'):
    if out_dtype is None or out_dtype == '':
        out_dtype = hcl.Int()
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

    if bitwidth == None:
        bitwidth = int(Input.dtype.split("int")[-1])
    batch, in_channel, in_height, in_width = Input.shape
    num_filter, filter_channel, kernel_h, kernel_w = Filter.shape
    assert in_channel == filter_channel
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify(
        (in_height - dilated_kernel_h + pad_top + pad_down) //
        stride_h + 1)
    out_width = simplify(
        (in_width - dilated_kernel_w + pad_left + pad_right) //
        stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    # temp = pad(Input, pad_before, pad_after, name=name+"_pad")
    temp = pad_(Input, name=name+"_pad", dtype=hcl.UInt(bitwidth))
    pad_in_height = in_height + pad_top + pad_down
    pad_in_width = in_width + pad_left + pad_right
    rc = hcl.reduce_axis(0, in_channel, name=name+'_rc')
    ry = hcl.reduce_axis(0, kernel_h, name=name+'_ry')
    rx = hcl.reduce_axis(0, kernel_w, name=name+'_rx')
    rb = hcl.reduce_axis(0, bitwidth, name=name+'_rb')
    # assert stride_h == 1 and stride_w == 1
    assert dilation_h == 1 and dilation_w == 1
    kernel_size = kernel_h * kernel_w
    if bitwidth == 1:
        const = 1
    elif bitwidth == 8:
        const = 0xff
    elif bitwidth == 16:
        const = 0xffff
    elif bitwidth == 32:
        const = 0xffffffff
    elif bitwidth == 64:
        const = 0xffffffffffffffff
    if threshold == None:
        rc_ = rc if in_channel != 1 else 0
        if mac:
            out = hcl.compute(
                (batch, out_channel, out_height, out_width),
                lambda nn, ff, yy, xx: hcl.sum(
                    hcl.select(
                        if_mac(yy*stride_h+ry, xx*stride_w+rx, pad_in_height, pad_in_width, pad_top, pad_left, pad_down, pad_right), # neglect padding pixels in mac
                        ((const - (temp[nn, rc_, yy * stride_h + ry, xx * stride_w + rx] ^ Filter[ff, rc_, ry, rx]))[rb] << 1) - 1,
                        0),
                    axis=[rc, ry, rx, rb], dtype=out_dtype, name=name+"_sum"),
                name=name,
                dtype=out_dtype)
        else:
            out = hcl.compute(
                (batch, out_channel, out_height, out_width),
                lambda nn, ff, yy, xx: kernel_size * bitwidth * in_channel - (
                    hcl.sum(
                        (temp[nn, rc_, yy * stride_h + ry, xx * stride_w + rx] ^ Filter[ff, rc_, ry, rx])[rb],
                    axis=[rc, ry, rx, rb], dtype=out_dtype, name=name+"_sum") << 1),
                name=name,
                dtype=out_dtype)
    else:
        bitwidth = out_channel
        rc_ = rc if in_channel != 1 else 0
        def genpack(nn, ff, yy, xx):
            out = hcl.scalar(0, name=name+"_pack", dtype=hcl.UInt(bitwidth))
            with hcl.for_(0, bitwidth) as k:
                out[0][(k+1) : k] = hcl.select(
                    hcl.sum(hcl.select(
                        if_mac(yy*stride_h+ry, xx*stride_w+rx, pad_in_height, pad_in_width, pad_top, pad_left, pad_down, pad_right), # neglect padding pixels in mac
                        ((const - (temp[nn, rc_, yy * stride_h + ry, xx * stride_w + rx] ^ Filter[ff*bitwidth+k, rc_, ry, rx]))[rb] << 1) - 1,
                    0), axis=[rc, ry, rx, rb], dtype=out_dtype, name=name+"_sum")
                    > threshold[ff*bitwidth+k, yy, xx],
                    1, 0)
            return out[0]
        return hcl.compute((batch, out_channel // bitwidth, out_height, out_width),
                            genpack, name=name, dtype=hcl.UInt(bitwidth))
    return out

def packed_conv2d_nhwc(
        Input,
        Filter,
        strides=[1, 1],
        padding=[0, 0],
        dilation=[1, 1],
        out_dtype=None,
        bitwidth=None,
        name='packed_binary_conv2d'):
    if out_dtype is None or out_dtype == '':
        out_dtype = hcl.Int()
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

    if bitwidth == None:
        bitwidth = int(Input.dtype.split("int")[-1])
    batch, in_height, in_width, in_channel = Input.shape
    num_filter, kernel_h, kernel_w, filter_channel = Filter.shape
    assert in_channel == filter_channel
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = simplify(
        (in_height - dilated_kernel_h + pad_top + pad_down) //
        stride_h + 1)
    out_width = simplify(
        (in_width - dilated_kernel_w + pad_left + pad_right) //
        stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = pad_nhwc(Input, name=name+"_pad", dtype=hcl.UInt(bitwidth))
    pad_in_height = in_height + pad_top + pad_down
    pad_in_width = in_width + pad_left + pad_right
    rc = hcl.reduce_axis(0, in_channel, name=name+'_rc')
    ry = hcl.reduce_axis(0, kernel_h, name=name+'_ry')
    rx = hcl.reduce_axis(0, kernel_w, name=name+'_rx')
    rb = hcl.reduce_axis(0, bitwidth, name=name+'_rb')
    # assert stride_h == 1 and stride_w == 1
    assert dilation_h == 1 and dilation_w == 1
    kernel_size = kernel_h * kernel_w
    if bitwidth == 1:
        const = 1
    elif bitwidth == 8:
        const = 0xff
    elif bitwidth == 16:
        const = 0xffff
    elif bitwidth == 32:
        const = 0xffffffff
    elif bitwidth == 64:
        const = 0xffffffffffffffff
    rc_ = rc if in_channel != 1 else 0
    out = hcl.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: hcl.sum(
            hcl.select(
                if_mac(yy*stride_h+ry, xx*stride_w+rx, pad_in_height, pad_in_width, pad_top, pad_left, pad_down, pad_right), # neglect padding pixels in mac
                ((const - (temp[nn, yy * stride_h + ry, xx * stride_w + rx, rc_] ^ Filter[ff, ry, rx, rc_]))[rb] << 1) - 1,
                0),
            axis=[ry, rx, rc, rb], dtype=out_dtype, name=name+"_sum"),
            name=name,
            dtype=out_dtype)
    return out

def max_pool2d_nchw(
        data,
        pooling=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        layout='NCHW',
        name='binary_max_pool2d'):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    max = hcl.reducer(
        hcl.min_value(data.dtype),
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
    if (pad_top,pad_left,pad_bottom,pad_right) != (0,0,0,0):
        data = pad(data, pad_before, pad_after, pad_value=hcl.min_value(data.dtype),name=name+"_pad")
    out_height = simplify(
        (height - pooling_h + pad_top + pad_bottom) // stride_h + 1)
    out_width = simplify(
        (width - pooling_w + pad_left + pad_right) // stride_w + 1)
    dheight = hcl.reduce_axis(0, pooling_h)
    dwidth = hcl.reduce_axis(0, pooling_w)
    return hcl.compute(
        (batch, channel, out_height, out_width),
        lambda i, c, h, w: hcl.select(max(data[i, c, h *
                                    stride_h +
                                    dheight, w *
                                    stride_w +
                                    dwidth], axis=[dheight, dwidth]) > 0,
                                    1,
                                    0),
        name=name, dtype=qtype_bit)

def packed_max_pool2d_nchw(
        data,
        pooling=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        layout='NCHW',
        name='packed_binary_max_pool2d',
        unpack=True):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    assert pooling == [2,2], "only support [2,2] padding now"
    max = hcl.reducer(
        hcl.min_value(data.dtype),
        lambda x, y: tvm.make.Max(x, y),
        data.dtype)
    pooling_h, pooling_w = pooling
    stride_h, stride_w = stride
    batch, channel, height, width = data.shape
    bitwidth = int(data.dtype.split("int")[-1])
    if len(padding) == 4:
        pad_top, pad_left, pad_bottom, pad_right = padding
    else:
        pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (pooling_h, pooling_w))
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_bottom, pad_right]
    if (pad_top,pad_left,pad_bottom,pad_right) != (0,0,0,0):
        data = pad(data, pad_before, pad_after, pad_value=hcl.min_value(data.dtype),name=name+"_pad")
    out_height = simplify(
        (height - pooling_h + pad_top + pad_bottom) // stride_h + 1)
    out_width = simplify(
        (width - pooling_w + pad_left + pad_right) // stride_w + 1)
    dheight = hcl.reduce_axis(0, pooling_h)
    dwidth = hcl.reduce_axis(0, pooling_w)
    reduce_or = hcl.reducer(0,lambda x, y: x | y, hcl.UInt(bitwidth))
    maxpool = hcl.compute(
        (batch, channel, out_height, out_width),
        lambda i, c, h, w: reduce_or(data[i, c, h * stride_h + dheight, w * stride_w + dwidth], axis=[dheight, dwidth]),
        name=name, dtype=hcl.UInt(bitwidth))
    if not unpack:
        return maxpool
    else:
        return hcl.compute((batch, channel * bitwidth, out_height, out_width),
            lambda i, c, h, w:
                maxpool[i, c // bitwidth, h, w][c % bitwidth],
            name=name+"_unpack",
            dtype=qtype_bit)

def packed_max_pool2d_nhwc(
        data,
        pooling=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        layout='NCHW',
        name='packed_binary_max_pool2d',
        unpack=True):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    assert pooling == [2,2], "only support [2,2] pooling now"
    assert padding == [0,0], "only support [0,0] padding now"
    max = hcl.reducer(
        hcl.min_value(data.dtype),
        lambda x, y: tvm.make.Max(x, y),
        data.dtype)
    pooling_h, pooling_w = pooling
    stride_h, stride_w = stride
    batch, height, width, channel = data.shape
    bitwidth = int(data.dtype.split("int")[-1])
    if len(padding) == 4:
        pad_top, pad_left, pad_bottom, pad_right = padding
    else:
        pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (pooling_h, pooling_w))
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_bottom, pad_right]
    if (pad_top,pad_left,pad_bottom,pad_right) != (0,0,0,0):
        data = pad(data, pad_before, pad_after, pad_value=hcl.min_value(data.dtype),name=name+"_pad")
    out_height = simplify(
        (height - pooling_h + pad_top + pad_bottom) // stride_h + 1)
    out_width = simplify(
        (width - pooling_w + pad_left + pad_right) // stride_w + 1)
    dheight = hcl.reduce_axis(0, pooling_h)
    dwidth = hcl.reduce_axis(0, pooling_w)
    reduce_or = hcl.reducer(0,lambda x, y: x | y, hcl.UInt(bitwidth))
    maxpool = hcl.compute(
        (batch, out_height, out_width, channel),
        lambda i, h, w, c: reduce_or(data[i, h * stride_h + dheight, w * stride_w + dwidth, c], axis=[dheight, dwidth]),
        name=name, dtype=hcl.UInt(bitwidth))
    return maxpool

def packed_max_pool2d_LB(
        data,
        pooling=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        name='packed_binary_max_pool2d_LB',):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    assert pooling == [2,2], "only support [2,2] padding now"
    pooling_h, pooling_w = pooling
    stride_h, stride_w = stride
    batch, channel, height, width = data.shape
    bitwidth = int(data.dtype.split("int")[-1])
    if len(padding) == 4:
        pad_top, pad_left, pad_bottom, pad_right = padding
    else:
        pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (pooling_h, pooling_w))
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_bottom, pad_right]
    if (pad_top,pad_left,pad_bottom,pad_right) != (0,0,0,0):
        data = pad(data, pad_before, pad_after, pad_value=hcl.min_value(data.dtype),name=name+"_pad")
    out_height = simplify(
        (height - pooling_h + pad_top + pad_bottom) // stride_h + 1)
    out_width = simplify(
        (width - pooling_w + pad_left + pad_right) // stride_w + 1)
    dtype = data.dtype
    maxpool = hcl.compute((batch, channel, out_height, out_width), lambda i, c, h, w: 0, name+"_res", dtype)
    LB = hcl.compute((2, width), lambda x, y: 0, name+"_LB", dtype)
    def loop_body(ii, cc, hh):
        with hcl.for_(0, 2, name=name+"_LB_i") as LB_i:
            with hcl.for_(0, width, name=name+"_LB_j") as LB_j:
                LB[LB_i, LB_j] = data[ii, cc, hh * 2 + LB_i, LB_j]
        with hcl.for_(0, out_width) as ww:
            val = hcl.scalar(0, name+"_val", dtype=dtype)
            with hcl.for_(0, 2, name=name+"_ry") as ry:
                with hcl.for_(0, 2, name=name+"_rx") as rx:
                    val.v |= LB[ry, ww * 2 + rx]
            maxpool[ii, cc, hh, ww] = val.v
    hcl.mutate((batch, channel, out_height), lambda ii, cc, hh: loop_body(ii, cc, hh), name)
    return maxpool

def packed_max_pool2d_nhwc_LB(
        data,
        pooling=[1, 1],
        stride=[1, 1],
        padding=[0, 0],
        name='packed_binary_max_pool2d_LB',):
    assert len(data.shape) == 4, "only support 4-dim pooling"
    assert len(stride) == 2, "only support 2-dim stride"
    assert pooling == [2,2], "only support [2,2] pooling now"
    assert padding == [0,0], "only support [0,0] padding now"
    pooling_h, pooling_w = pooling
    stride_h, stride_w = stride
    batch, height, width, channel = data.shape
    bitwidth = int(data.dtype.split("int")[-1])
    if len(padding) == 4:
        pad_top, pad_left, pad_bottom, pad_right = padding
    else:
        pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (pooling_h, pooling_w))
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_bottom, pad_right]
    if (pad_top,pad_left,pad_bottom,pad_right) != (0,0,0,0):
        data = pad(data, pad_before, pad_after, pad_value=hcl.min_value(data.dtype),name=name+"_pad")
    out_height = simplify(
        (height - pooling_h + pad_top + pad_bottom) // stride_h + 1)
    out_width = simplify(
        (width - pooling_w + pad_left + pad_right) // stride_w + 1)
    dtype = data.dtype
    maxpool = hcl.compute((batch, out_height, out_width, channel), lambda i, h, w, c: 0, name+"_res", dtype)
    LB = hcl.compute((2, width), lambda x, y: 0, name+"_LB", dtype)
    assert channel == 1, "only support 1 packed channel"
    def loop_body(ii, hh):
        with hcl.for_(0, 2, name=name+"_LB_i") as LB_i:
            with hcl.for_(0, width, name=name+"_LB_j") as LB_j:
                LB[LB_i, LB_j] = data[ii, hh * 2 + LB_i, LB_j, 0]
        with hcl.for_(0, out_width) as ww:
            val = hcl.scalar(0, name+"_val", dtype=dtype)
            with hcl.for_(0, 2, name=name+"_ry") as ry:
                with hcl.for_(0, 2, name=name+"_rx") as rx:
                    val.v |= LB[ry, ww * 2 + rx]
            maxpool[ii, hh, ww, 0] = val.v
    hcl.mutate((batch, out_height), lambda ii, hh: loop_body(ii, hh), name)
    return maxpool

def batch_norm(
        data,
        gamma,
        beta,
        moving_mean,
        moving_var,
        M0=1,
        axis=1,
        epsilon=10**-5,
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

    var_w = np.sqrt(2. / (9. * M0)) # predefined constant
    out = hcl.compute(data.shape, lambda *x: hcl.select(
                    (data[x] * var_w - moving_mean[get_axis(axis, x)]) /
                    (hcl.sqrt(moving_var[get_axis(axis, x)] + epsilon)) * gamma[get_axis(axis, x)]
                    + beta[get_axis(axis, x)] > 0,
                    1, # quantize
                    0), name=name, dtype=qtype_bit)
    return out, moving_mean, moving_var

def batch_norm_threshold(
        data,
        threshold,
        name="batch_norm_threshold"):
    return hcl.compute(data.shape, lambda i, c, h, w: hcl.select(
                    data[i, c, h, w] > threshold[c, h, w],
                    1, # quantize
                    0), name=name, dtype=qtype_bit)

def packed_batch_norm_threshold(
        data,
        threshold,
        name="packed_batch_norm_threshold"):
    batch, channel, out_height, out_width = data.shape
    bitwidth = channel # pack channels
    def genpack(i, c, h, w):
        out = hcl.scalar(0, name=name+"_pack", dtype=hcl.UInt(bitwidth))
        with hcl.for_(0, bitwidth) as k:
            out[0][(k+1) : k] = hcl.select(data[i, c*bitwidth+k, h, w] > threshold[c*bitwidth+k, h, w], 1, 0)
        return out[0]
    return hcl.compute((batch, channel//bitwidth, out_height, out_width),
                        genpack, name=name, dtype=hcl.UInt(bitwidth))

def packed_batch_norm_threshold_nhwc(
        data,
        threshold,
        name="packed_batch_norm_threshold"):
    batch, out_height, out_width, channel = data.shape
    bitwidth = channel # pack channels
    def genpack(i, h, w, c):
        out = hcl.scalar(0, name=name+"_pack", dtype=hcl.UInt(bitwidth))
        with hcl.for_(0, bitwidth) as k:
            out[0][(k+1) : k] = hcl.select(data[i, h, w, c*bitwidth+k] > threshold[h, w, c*bitwidth+k], 1, 0)
        return out[0]
    return hcl.compute((batch, out_height, out_width, channel//bitwidth),
                        genpack, name=name, dtype=hcl.UInt(bitwidth))