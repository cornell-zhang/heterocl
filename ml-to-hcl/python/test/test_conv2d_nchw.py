import heterocl as hcl
import numpy as np
import hlib
hcl.init()

I = hcl.placeholder((1,3,3,1),"I")
F = hcl.placeholder((3,1,1,1),  "F")
B = hcl.placeholder((3,), "B"      )

def _conv2d_nhwc(Input, Filter, Bias=None, stride=[1,1], padding=[1,1], dilation=[1,1], name='conv2d', out_dtype=None ):
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
    pad_top, pad_left, pad_down, pad_right = hlib.nn.get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_height = hlib.nn.simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = hlib.nn.simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0,pad_top, pad_left,0]
    pad_after = [0,pad_down, pad_right,0]
    print(pad_before,pad_after)
    temp = hlib.nn.pad(Input, pad_before, pad_after, name="pad_temp")
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
              axis=[ry, rx, rc]), name=name,)
          #attrs=OrderedDict([
          #    ('p', kernel_h),
          #    ('q', kernel_w),
          #    ('in_num', in_channel),
          #    ('out_num', out_channel),
          #    ('out_img_w', out_width),
          #    ('out_img_h', out_height),
          #    ('cin_dtype', tvm.make.StringImm(Input.dtype)),
          #    ('filter_dtype', tvm.make.StringImm(Filter.dtype))]))



s = hcl.create_schedule([I,F,B],_conv2d_nhwc)
f = hcl.build(s)

data = np.random.randint(5,size=(1,3,3,1))
filt = np.array([[[[1]]],[[[2]]],[[[3]]]])
bias = np.array([0,0,0])
print(data.shape,filt.shape,bias.shape)
_out = hcl.asarray(np.zeros((1,5,5,3)))
data = hcl.asarray(data)
filt = hcl.asarray(filt)
bias = hcl.asarray(bias)
f(data,filt,bias,_out)
print(data.asnumpy())
print(np.transpose(_out.asnumpy(),(0,3,2,1)))
