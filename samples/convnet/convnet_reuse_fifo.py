import heterocl as hcl
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hw', default=False, action='store_true')
parser.add_argument('--optimize', default=False, action='store_true')

# Conv2d NCHW. It can be changed to NHWC easily
def conv2d(img, weight, name="conv"):
    assert img.shape[0] == weight.shape[1]
    IC, H, W = img.shape
    OC, _, R, C = weight.shape
    OH = H - R + 1
    OW = W - C + 1

    # Reduction loops
    rc = hcl.reduce_axis(0, IC)
    ry = hcl.reduce_axis(0, R)
    rx = hcl.reduce_axis(0, C)

    return hcl.compute(
        (OC, OH, OW),
        lambda ff, yy, xx: hcl.sum(
            img[rc, yy + ry, xx + rx] * weight[ff, rc, ry, rx],
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
    return hcl.compute(op.shape, 
        lambda *args: hcl.select(op[args] > 0, op[args], 0), name=name)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def ConvNet(hw_exe=False, optimize=False):
    dtype=hcl.Float()
    hcl.init(dtype)

    input_size = (1,30,30)
    img = hcl.placeholder(input_size, dtype=dtype, name="input_image")
    conv_w1 = hcl.placeholder((16,1,3,3), dtype=dtype, name="conv_w1")  # weight for conv
    conv_w2 = hcl.placeholder((64,16,3,3), dtype=dtype, name="conv_w2")  # weight for conv
    dense_w = hcl.placeholder((64*26*26,10), dtype=dtype, name="dense_w") # weight for dense

    # A three layer ConvNet example 
    def top(img, conv_w1, conv_w2, dense_w):
        output1 = conv2d(img, conv_w1, name="conv1")
        output2 = conv2d(output1, conv_w2, name="conv2")
        output3 = reshape(relu(output2, name="relu"), name="reshape") # output2 is the reshapeed tensor
        return dense(output3, dense_w, name="dense")  # return one-hot pred (1,10)

    s = hcl.create_schedule([img, conv_w1, conv_w2, dense_w], top)

    # Offload the main body to FPGA
    p = hcl.Platform.xilinx_u280
    s.to([top.conv1, conv_w2, dense_w], p.xcel)
    s.to(top.dense, p.host)

    if optimize:
        print("[  HCL  ] Adding reuse buffer and FIFO channels...")
        # Create reuse buffers for conv2d layer
        LB = s.reuse_at(top.conv1, s[top.conv2], top.conv2.axis[1], "LB")
        WB = s.reuse_at(LB,  s[top.conv2], top.conv2.axis[2], "WB")
        s.partition(WB, hcl.Partition.Complete, dim=0)
        s.partition(LB, hcl.Partition.Complete, dim=2)

        # Connect layers with FIFOs
        s.to(top.conv2, top.relu, depth=32)
        s.to(top.relu, top.reshape, depth=32)
        s.to(top.reshape, top.dense, depth=32)

    if hw_exe: 
        print("[  HCL  ] Starting execution on real FPGA hardware...")
        p.config(compile="vitis", mode="hw_exe", project="hcl_prj_reuse_hw_exe")
    else:     
        print("[  HCL  ] Synthesizing hardware with Vivado HLS...") 
        p.config(compile="vivado_hls", mode="csyn", project="hcl_prj_reuse_hls")

    f = hcl.build(s, target=p)

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

    args = list()
    args.append(x_test[0].reshape(1,30,30))

    args.append(conv_w1)
    args.append(conv_w2)
    args.append(dense_w)
    args.append(np.zeros(shape=(10,)))
    
    # Generate HLS code
    f.inspect(args)
    f.execute(args)

    # Print HLS report 
    if not hw_exe:
      f.report()

if __name__ == "__main__":
    args = parser.parse_args()
    ConvNet(args.hw, args.optimize)

    

