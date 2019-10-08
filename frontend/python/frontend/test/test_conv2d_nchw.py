import heterocl as hcl
import numpy as np
import heterocl.tvm as tvm
import hlib
import numpy.testing as tst
from hlib.nn import get_pad_tuple
hcl.init(hcl.Float())


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
            strides=stride,
            padding=padding,
            dilation=dilation,
            data_layout=layout)
    def _conv2d_old(
            I,
            F,
            stride=stride,
            padding=padding):
        return hlib.nn.conv2d_nchw_old(
            I,
            F,
            strides=stride,
            padding=padding,
            dilation=dilation,)
    s1 = hcl.create_schedule([I, F], _conv2d)
    s2 = hcl.create_schedule([I, F], _conv2d)
    f1 = hcl.build(s1)
    f2 = hcl.build(s2)

    data = hcl.asarray(np.random.rand(*in_size))
    filt = hcl.asarray(np.random.rand(*filt_size))

    def output_size(in_size, filt_size, padding, dilation, stride, layout):
        if layout == "NHWC":
            B, I_h, I_w, I_c = in_size
            K_h, K_w, F_c, n_filt = filt_size
        if layout == "NCHW":
            B, I_c, I_h, I_w = in_size
            n_filt, F_c, K_h, K_w = filt_size
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
        if layout == "NCHW":
            return (B, O_c, O_h, O_w)
    size = output_size(
                in_size,
                filt_size,
                padding,
                dilation,
                stride,
                layout)
    _out1 = hcl.asarray(np.zeros(size))
    _out2 = hcl.asarray(np.zeros(size))
    f1(data,filt, _out1)
    f2(data,filt,_out2)
    tst.assert_almost_equal(_out1.asnumpy(),_out2.asnumpy())
    #return(data.asnumpy(), filt.asnumpy(), np.transpose(_out1.asnumpy(), (0, 1, 2, 3)),np.transpose(_out2.asnumpy(), (0, 1, 2, 3)))

conv2d_test((3, 1, 3, 3), (3, 3, 3, 1), stride=[1, 1], padding=[0, 0], dilation=[1, 1], layout="NCHW")
conv2d_test((3, 1, 1, 1), (3, 1, 1, 1), stride=[1, 1], padding=[0, 0], dilation=[1, 1], layout="NCHW")
conv2d_test((3, 1, 3, 3), (3, 3, 3, 1), stride=[2, 2], padding=[0, 0], dilation=[1, 1], layout="NCHW")
conv2d_test((3, 1, 3, 3), (3, 3, 3, 1), stride=[1, 1], padding=[1, 1], dilation=[1, 1], layout="NCHW")
conv2d_test((5, 5, 5, 5), (1, 1, 1, 1), stride=[1, 1], padding=[0, 0], dilation=[2, 2], layout="NCHW")
conv2d_test((10, 1, 3, 3), (10, 3, 3, 1), stride=[2, 2], padding=[2, 1], dilation=[1, 1], layout="NCHW")
conv2d_test((255, 3, 32, 32), (10, 3, 3, 3), stride=[2, 2], padding=[2, 1], dilation=[1, 1], layout="NCHW")
