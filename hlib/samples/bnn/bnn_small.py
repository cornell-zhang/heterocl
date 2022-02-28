import heterocl as hcl
import hlib.bnn as bnn
import numpy as np
import sys

target = hcl.Platform.xilinx_zc706
target.config(compiler="vivado_hls", mode="csim|csyn", project="bnn.prj")
test_size = 100
batch_size = 100
qtype_bit = hcl.UInt(1)  # weights
qtype_int = hcl.Int(6)  # not unsigned!
qtype_float = hcl.Fixed(20, 10)
qtype_packed = hcl.UInt(32)


def build_bnn(input_image, w_conv1, bn_t1,
              w_conv2, bn_t2,
              w_fc1, b_fc1,
              w_fc2, b_fc2):  # 1*16*16
    conv1 = bnn.conv2d_nchw(input_image, w_conv1, padding=[
                            1, 1], name="conv1", out_dtype=qtype_int)  # 16*16*16
    bn1 = bnn.batch_norm_threshold(conv1, bn_t1, name="bn1")
    maxpool1 = bnn.max_pool2d_nchw(
        bn1, [2, 2], [2, 2], name="maxpool1")  # 16*8*8
    conv2 = bnn.conv2d_nchw(maxpool1, w_conv2, padding=[
                            1, 1], name="conv2", out_dtype=qtype_int)  # 32*8*8
    bn2 = bnn.batch_norm_threshold(conv2, bn_t2, name="bn2")
    maxpool2 = bnn.max_pool2d_nchw(
        bn2, [2, 2], [2, 2], name="maxpool2")  # 32*4*4=512
    flat = bnn.flatten(maxpool2, name="flatten")
    fc1 = bnn.dense(flat, w_fc1, b_fc1, True, name="fc1",
                    dtype=qtype_float)  # 512->256
    fc2 = bnn.dense(fc1, w_fc2, b_fc2, False, name="fc2")  # 256->10
    return fc2


def build_bnn_inf(batch_size, target):
    hcl_ph = []
    input_image = hcl.placeholder(
        (batch_size, 1, 16, 16), "input_image", qtype_bit)
    hcl_ph.append(hcl.placeholder((16, 1, 3, 3), "w_conv1", qtype_bit))
    hcl_ph.append(hcl.placeholder((16, 16, 16), "bn_t1", qtype_float))
    hcl_ph.append(hcl.placeholder((32, 16, 3, 3), "w_conv2", qtype_bit))
    hcl_ph.append(hcl.placeholder((32, 8, 8), "bn_t2", qtype_float))
    hcl_ph.append(hcl.placeholder((256, 512), "w_fc1", qtype_bit))
    hcl_ph.append(hcl.placeholder((256,), "b_fc1", qtype_float))
    hcl_ph.append(hcl.placeholder((10, 256), "w_fc2", qtype_bit))
    hcl_ph.append(hcl.placeholder((10,), "b_fc2", qtype_float))
    # for name in params:
    #     dtype = qtype_bit if ("conv" in name or "w_" in name) else qtype_float
    #     hcl_ph.append(hcl.placeholder(params[name].shape,name,dtype=dtype))

    # build the network
    s = hcl.create_schedule([input_image] + hcl_ph, build_bnn)
    print(s.device_module)

    if isinstance(target, hcl.Platform):
        s.to([input_image] + hcl_ph, target.xcel)
        s.to(build_bnn.fc2, target.host)

    return hcl.build(s, target=target)


if __name__ == '__main__':

    f = build_bnn_inf(batch_size, target)
    f()
