import heterocl as hcl
import numpy as np
import sys

# Conv2d NCHW. It can be changed to NHWC easily
def conv2d(img, weight, paddings=[0,0], strides=[1,1], name="conv"):
    assert img.shape[0] == weight.shape[1]
    IC, H, W = img.shape
    OC, IC, R, C = weight.shape
    stride_h, stride_w = strides
    padding_h, padding_w = paddings
    OH = H + padding_h - R + 1
    OW = W + padding_w - C + 1

    # reduction loops
    rc = hcl.reduce_axis(0, IC)
    ry = hcl.reduce_axis(0, R)
    rx = hcl.reduce_axis(0, C)

    return hcl.compute(
        (OC, OH, OW),
        lambda ff, yy, xx: hcl.sum(
            img[rc, yy * stride_h + ry, xx * stride_w + rx] *
            weight[ff, rc, ry, rx],
            axis=[rc, ry, rx],
            dtype=hcl.Float()),
        name=name)

def dense(A, B, name="dense"):
    assert B.shape[0] == A.shape[1]
    m, k = A.shape
    _, n = B.shape
    r = hcl.reduce_axis(0, k)
    return hcl.compute((m, n), 
        lambda x, y: hcl.sum(A[x,r]*B[r,y], axis=[r], dtype=hcl.Float()), 
            dtype=A.dtype, name=name)

# Flatten 3d tensor into 1d
def flatten(op, name="flatten"):
    new_shape = (1, np.prod(op.shape))
    new_tensor = hcl.compute(new_shape, lambda *args: 0, name=name)
    with hcl.Stage(name):
        with hcl.for_(0, op.shape[0]) as d2:
            with hcl.for_(0, op.shape[1]) as d1:
                with hcl.for_(0, op.shape[2]) as d0:
                    new_tensor[0][(d2 * op.shape[1] + d1) * op.shape[2] + d0] = op[d2,d1,d0]
    return new_tensor

def relu(op, name="relu"):
    @hcl.def_([()])
    def select(A):
        temp = hcl.scalar(A)
        hcl.return_(hcl.select(temp > 0.0, temp, 0.0))
    return hcl.compute(op.shape, 
        lambda *args: select(op[args]), name=name)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def ConvNet():
    dtype=hcl.Float()
    hcl.init(dtype)

    p = hcl.Platform.aws_f1
    p.config(compile="vitis", mode="sw_sim", project="hcl_prj_quant")

    input_size = (1,30,30)
    img = hcl.placeholder(input_size, dtype=dtype, name="input_image")
    conv_w1 = hcl.placeholder((16,1,3,3), dtype=dtype, name="conv_w1")  # weight for conv
    conv_w2 = hcl.placeholder((64,16,3,3), dtype=dtype, name="conv_w2")  # weight for conv
    dense_w = hcl.placeholder((64*26*26,10), dtype=dtype, name="dense_w") # weight for dense

    # A two layer ConvNet example 
    def top(img, conv_w1, conv_w2, dense_w):
        output1 = conv2d(img, conv_w1, name="conv1")
        output2 = conv2d(output1, conv_w2, name="conv2")
        output3 = flatten(relu(output2, name="relu"), name="flatten") # output2 is the flattened tensor
        final = dense(output3, dense_w, name="dense")  # return one-hot pred (1,10)
        return final

    # Data tyepe customization
    dtype_quant = hcl.Float()
    scheme = hcl.create_scheme([img, conv_w1, conv_w2, dense_w], top)
    # scheme.quantize([top.conv1, top.conv2, top.dense, top.relu, top.flatten], dtype_quant)
    s = hcl.create_schedule_from_scheme(scheme)

    p = hcl.Platform.aws_f1
    p.config(compile="vitis", mode="sw_sim", project="baseline")
    # f = hcl.build(s, target=p)
    f = hcl.build(s, target="llvm")

    # weights loading from npy
    with open('convnet.npy', 'rb') as fp:
        # the weight matrix exported from keras is reversed
        # https://stackoverflow.com/a/46757884/13411736
        w1 = np.load(fp); w2 = np.load(fp); w3 = np.load(fp)
        conv_w1 = hcl.asarray(np.transpose(w1,(4,3,0,1,2)).reshape(16,1,3,3))
        conv_w2 = hcl.asarray(np.transpose(w2,(4,3,0,1,2)).reshape(64,16,3,3))
        dense_w = hcl.asarray(w3.reshape(26*26*64,10))

    # verify the accuracy
    with open('input_data.npy', 'rb') as fp:
        x_test = np.load(fp)
        y_test = np.load(fp)
    
    # test the first data
    with open('intermediate.npy', 'rb') as fp:
        out_conv1 = np.load(fp)
        out_conv2 = np.load(fp)
        out_flatten = np.load(fp)
        out_dense = np.load(fp)

    match = 0
    for index in range(1000):
        in_data = x_test[index]
        in_data = hcl.asarray(in_data.reshape(1,30,30))

        output  = hcl.asarray(np.ones((10,1)), dtype=dtype_quant)
        f(in_data, conv_w1, conv_w2, dense_w, output)
        scores = output.asnumpy()
        if np.argmax(softmax(scores)) == np.argmax(y_test[index]):
          match += 1
    acc = match /10
    print(f"ACC {acc}%")

if __name__ == "__main__":
    ConvNet()

    

