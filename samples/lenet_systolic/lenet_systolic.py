import heterocl as hcl
import hlib
import numpy as np
from flexcnn_conv import conv2d_nchw_systolic

hcl.init(hcl.Float(32))

def softmax(out, x):
    assert len(x.shape) == 2, "only support 2-dim softmax"
    m, n = x.shape
    k = hcl.reduce_axis(0, n)
    max_elem = hcl.compute((m,), lambda i: hcl.max(x[i, k], axis=k))
    k = hcl.reduce_axis(0, n)
    expsum = hcl.compute((m,),
            lambda i: hcl.sum(hcl.exp(x[i, k] - max_elem[i]), axis=k))
    return hcl.update(out,
            lambda i, j: hcl.exp(x[i, j] - max_elem[i]) / expsum[i])

def build_lenet(input_image, weight_conv1, weight_conv2,
                weight_fc1, weight_fc2, lenet):
    # first conv
    # conv1 = hlib.op.nn.conv2d_nchw(input_image, weight_conv1)
    conv1 = conv2d_nchw_systolic(input_image, weight_conv1)
    tanh1 = hlib.op.math.tanh(conv1, "tanh1")
    pool1 = hlib.op.nn.max_pool(tanh1, kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = hlib.op.nn.conv2d_nchw(pool1, weight_conv2)
    tanh2 = hlib.op.math.tanh(conv2, "tanh2")
    pool2 = hlib.op.nn.max_pool(tanh2, kernel=(2,2), stride=(2,2))
    # first fc
    flat = hlib.op.nn.flatten(pool2)
    fc1 = hlib.op.nn.dense(flat, weight_fc1)
    tanh3 = hlib.op.math.tanh(fc1, "tanh3")
    # second fc
    fc2 =  hlib.op.nn.dense(tanh3, weight_fc2)
    # loss
    return softmax(lenet, fc2)

def build_lenet_inf(batch_size=1000, target=None):
    # set up input/output placeholders
    input_image = hcl.placeholder((batch_size, 1, 28, 28), "input_image")
    weight_conv1 = hcl.placeholder((20, 1, 5, 5), "weight_conv1")
    weight_conv2 = hcl.placeholder((50, 20, 5, 5), "weight_conv2")
    weight_fc1 = hcl.placeholder((500, 800), "weight_fc1")
    weight_fc2 = hcl.placeholder((10, 500), "weight_fc2")
    lenet = hcl.placeholder((batch_size, 10), "lenet")
    s = hcl.create_schedule([input_image, weight_conv1, weight_conv2, weight_fc1, weight_fc2, lenet], build_lenet)
    return hcl.build(s, target=target)

if __name__ == "__main__":
    code3 = build_lenet_inf(target='vhls')
    with open('vhls_systolic_code.cpp', 'w') as f:
        f.write(code3)
