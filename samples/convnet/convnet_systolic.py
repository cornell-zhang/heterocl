import heterocl as hcl
import numpy as np

# Conv2d NCHW. It can be changed to NHWC easily
def conv2d(img, weight, paddings=[0,0], strides=[1,1], name="conv"):
    assert img.shape[-1] == weight.shape[-1]
    H, W, IC = img.shape
    OC, R, C, IC = weight.shape
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
        lambda ff, yy, xx: sum(
            img[rc, yy * stride_h + ry, xx * stride_w + rx] *
            weight[ff, rc, ry, rx],
            axis=[rc, ry, rx]),
        name=name)

def dense(A, B, name="dense"):
    assert B.shape[0] == A.shape[1]
    m, k = A.shape
    _, n = B.shape
    r = hcl.reduce_axis(0,k)
    return hcl.compute((m, n), 
        lambda x, y: hcl.sum(A[x,r]*B[r,y], axis=[r]), 
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

def ConvNet(stream=False):
    dtype=hcl.Float()
    hcl.init(dtype)

    input_size = (28,28,1); OC = 64
    img = hcl.placeholder(input_size, dtype=dtype, name="input_image")
    conv_w = hcl.placeholder((OC,3,3,1), dtype=dtype, name="conv_w")  # weight for conv
    dense_w = hcl.placeholder((26*26*OC,10), dtype=dtype, name="dense_w") # weight for dense

    # A two layer ConvNet example 
    def top(img, conv_w, dense_w):
        output1 = conv2d(img, conv_w, name="conv")
        output2 = flatten(relu(output1, name="relu"), name="flatten") # output2 is the flattened tensor
        return dense(output2, dense_w, name="dense")  # return one-hot pred (1,10)

    p = hcl.Platform.aws_f1
    p.config(compile="vitis", mode="sw_sim", project="Baseline") 

    # Data tyepe customization
    dtype_quant = hcl.Fixed(8,2)
    scheme = hcl.create_scheme([img, conv_w, dense_w], top)
    scheme.downsize([top.conv, top.dense, top.relu, top.flatten], dtype_quant)
    s = hcl.create_schedule_from_scheme(scheme)

    # Create systolic array for Conv and Dense layers
    s[top.conv].systolic()
    s[top.dense].systolic()

    # Stream first SA's output to relu and flatten
    s.to(top.conv, top.relu, depth=32)
    s.to(top.flatten, top.dense, depth=32)

    # Offload the main body to FPGA
    s.to([img, conv_w, dense_w], p.xcel)
    s.to(top.dense, p.host)

    f = hcl.build(s, target=p)
    args = list()

    # loading data from numpy serialized data
    low = 0; high = 10
    args.append(np.random.uniform(low=low, high=high, size=img.shape))

    # weights loading from npy
    with open('convnet.npy', 'rb') as fp:
        conv_w = np.load(fp)
        dense_w = np.load(fp)

    args.append(conv_w)
    args.append(dense_w)
    args.append(np.random.uniform(low=low, high=high, size=(10,)))
    
    # Generate code
    f.inspect(args)

if __name__ == "__main__":
    ConvNet()

    

