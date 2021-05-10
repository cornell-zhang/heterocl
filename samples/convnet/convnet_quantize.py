import heterocl as hcl
import numpy as np
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--quantize', default=False, action='store_true')

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

def reshape(op, name="reshape"):
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

def ConvNet(quantize=False):
    dtype=hcl.Float()
    hcl.init(dtype)

    input_size = (1,30,30)
    img = hcl.placeholder(input_size, dtype=dtype, name="input_image")
    conv_w1 = hcl.placeholder((16,1,3,3), dtype=dtype, name="conv_w1")  # weight for conv
    conv_w2 = hcl.placeholder((64,16,3,3), dtype=dtype, name="conv_w2")  # weight for conv
    dense_w = hcl.placeholder((64*26*26,10), dtype=dtype, name="dense_w") # weight for dense

    # A two layer ConvNet example 
    def top(img, conv_w1, conv_w2, dense_w):
        output1 = conv2d(img, conv_w1, name="conv1")
        output2 = conv2d(output1, conv_w2, name="conv2")
        output3 = reshape(relu(output2, name="relu"), name="reshape") # output2 is the reshapeed tensor
        final = dense(output3, dense_w, name="dense")  # return one-hot pred (1,10)
        return final

    # Data tyepe customization
    dtype_quant = hcl.Fixed(8,2)
    scheme = hcl.create_scheme([img, conv_w1, conv_w2, dense_w], top)
    if quantize:
      print("[  HCL  ] Quantizing the activation and conv2 layer...")
      scheme.quantize([top.relu, conv_w2], dtype_quant)
    s = hcl.create_schedule_from_scheme(scheme)

    # Build function from HCL schedule
    f = hcl.build(s, target="llvm")

    # Weights loading from npy
    with open('convnet.npy', 'rb') as fp:
        w1 = np.load(fp); w2 = np.load(fp); w3 = np.load(fp)
        conv_w1 = hcl.asarray(np.transpose(w1,(4,3,0,1,2)).reshape(16,1,3,3))
        conv_w2 = hcl.asarray(np.transpose(w2,(4,3,0,1,2)).reshape(64,16,3,3))
        dense_w = hcl.asarray(w3.reshape(26*26*64,10))

    # Verify the accuracy
    with open('input_data.npy', 'rb') as fp:
        x_test = np.load(fp)
        y_test = np.load(fp)
    
    match = 0
    count = 1200
    print("[  HCL  ] Running inference on validation dataset...")
    for index in range(count):
        in_data = x_test[index]
        in_data = hcl.asarray(in_data.reshape(1,30,30))

        output  = hcl.asarray(np.ones((10,1)), dtype=hcl.Float())
        f(in_data, conv_w1, conv_w2, dense_w, output)
        scores = output.asnumpy()
        if np.argmax(softmax(scores)) == np.argmax(y_test[index]):
          match += 1

    acc = (match / count) * 100
    print("[  HCL  ] MNIST accuracy %.2f" % round(acc, 2), "%")

if __name__ == "__main__":
    args = parser.parse_args()
    ConvNet(args.quantize)

    

