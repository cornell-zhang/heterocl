import heterocl as hcl
import numpy as np
import heterocl.tvm as tvm
import hlib
from hlib.nn import get_pad_tuple
hcl.init()


def conv2d_test(
    in_size, filt_size, stride=[
        1, 1], padding=[
            0, 0], dilation=[
                1, 1], layout="NHWC"):

    I = hcl.placeholder(in_size, "I")
    F = hcl.placeholder(filt_size, "F")

    def _conv2d(
            I,
            F,
            stride=stride,
            padding=padding,
            dilation=dilation,
            layout=layout):
        return hlib.nn.conv2d(
            I,
            F,
            stride=stride,
            padding=padding,
            dilation=dilation,
            layout=layout)
    s = hcl.create_schedule([I, F], _conv2d)
    f = hcl.build(s)

    data = hcl.asarray(np.random.randint(1, high=2, size=in_size))
    filt = hcl.asarray(np.random.randint(1, high=2, size=filt_size))

    def output_size(in_size, filt_size, padding, dilation, stride, layout):
        if layout == "NHWC":
            B, I_h, I_w, I_c = in_size
            K_h, K_w, F_c, n_filt = filt_size

        D_h, D_w = dilation
        S_h, S_w = stride
        dilated_K_h = (K_h - 1) * D_h + 1
        dilated_K_w = (K_w - 1) * D_w + 1
        P_t, P_l, P_d, P_r = get_pad_tuple(padding,
                                           (dilated_K_h, dilated_K_w))
        O_c = n_filt
        O_h = (I_h - dilated_K_h + P_t + P_d) // S_h + 1
        O_w = (I_w - dilated_K_w + P_l + P_r) // S_w + 1
        if layout == "NHWC":
            return (B, O_h, O_w, O_c)
    _out = hcl.asarray(
        np.zeros(
            output_size(
                in_size,
                filt_size,
                padding,
                dilation,
                stride,
                layout)))
    f(data, filt, _out)
    return(data.asnumpy(), filt.asnumpy(), np.transpose(_out.asnumpy(), (0, 3, 2, 1)))


in_size = (1, 3, 3, 1)
filt_size = (1, 1, 1, 1)
padding = [1, 1]
stride = [1, 1]
dilation = [1, 1]
layout = "NHWC"

print(conv2d_test(in_size, filt_size, stride, padding, dilation, layout))
