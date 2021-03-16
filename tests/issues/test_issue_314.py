import heterocl as hcl
import os
from collections import OrderedDict
import heterocl.tvm as tvm
import numpy as np

dtype = hcl.Float()
sum = hcl.reducer(0, lambda x, y: x + y, dtype)
max = hcl.reducer(-1, lambda x, y: tvm.make.Max(x, y), dtype)
def simplify(expr):
    return tvm.ir_pass.Simplify(expr) if isinstance(expr, tvm.expr.Expr) else expr

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

def test_conv_reuse(dtype=hcl.Int()):
    hcl.init(dtype)
    A = hcl.placeholder((1,1,8,8), "A", dtype)
    B = hcl.placeholder((1,1,3,3), "B", dtype)

    def two_conv(A, B):
        out1 = conv2d_nchw(A, B, name="conv1")
        return conv2d_nchw(out1, B, name="conv2")

    s = hcl.create_schedule([A, B], two_conv)
    target = hcl.Platform.aws_f1
    target.config(compile="vitis", mode="hw_sim")

    # Move compute to device
    s.to([A, B], target.xcel)
    s.to(two_conv.conv2, target.host)

    # Move data between two conv kernels
    # Automatically create reuse buffers to create sequential access
    s.to(two_conv.conv1, two_conv.conv2)

    np_A = np.random.randint(10, size=(1,1,8,8))
    np_B = np.random.randint(10, size=(1,1,3,3))
    np_C = np.zeros((1,1,4,4))
    args = (np_A, np_B, np_C)

    # Generate local projects
    print(hcl.lower(s))

if __name__ == "__main__":
    test_conv_reuse()

