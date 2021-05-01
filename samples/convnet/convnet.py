import heterocl as hcl
import numpy as np

def conv2d_nhwc(img, weight, padding=0, name="Y0"):
    assert img.shape[-1] == weight.shape[-1]
    H, W, IC = img.shape
    OC, R, C, IC = weight.shape
    OH = H + padding - R + 1
    OW = W + padding - C + 1

    out = hcl.compute((OH, OW, OC), lambda *args: 0, dtype=img.dtype, name=name)
    with hcl.Stage(f"Conv_{name}"):
        with hcl.for_(0, OC) as oc:
            with hcl.for_(0, OH) as oh:
                with hcl.for_(0, OW) as ow:
                    out[oh][ow][oc] = 0
                    with hcl.for_(0, IC) as ic:
                        with hcl.for_(0, R) as r:
                            with hcl.for_(0, C) as c:
                                out[oh][ow][oc] += img[oh+r][ow+c][ic] * weight[oc][r][c][ic]
    return out

def dense(A, B, name="Y0"):
    print("[DenseLayer] ", A.shape, B.shape)
    assert B.shape[0] == A.shape[1]
    m, k = A.shape
    _, n = B.shape
    Y = hcl.compute((m, n), lambda *args: 0, dtype=A.dtype, name=name)
    with hcl.Stage(f"Dense_{name}"):
        with hcl.for_(0, m) as i:
            with hcl.for_(0, n) as j:
                Y[i][j] = 0
                with hcl.for_(0, k) as r:
                    Y[i][j] += A[i][r] * B[r][j]   
    return Y 

# Flatten 3d tensor into 1d
def flatten(op, name="FL"):
    print("[FlattenLayer] ", op.shape)
    new_shape = (1, np.prod(op.shape))
    new_tensor = hcl.compute(new_shape, lambda *args: 0, name=name)

    with hcl.Stage(f"Flatten"):
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

    # Infer image in
    img = hcl.placeholder((28,28,16), dtype=dtype, name="input_image")
    conv_w = hcl.placeholder((64,3,3,16), dtype=dtype, name="conv_w")  # weight for conv
    dense_w = hcl.placeholder((26*26*64,10), dtype=dtype, name="dense_w") # weight for dense

    # A two layer ConvNet example 
    def top(img, conv_w, dense_w):
        output1 = conv2d_nhwc(img, conv_w, name="L1")
        output2 = flatten(relu(output1, name="relu")) # output2 is the flattened tensor
        return dense(output2, dense_w, name="L2")  # return one-hot pred (1,10)

    p = hcl.Platform.aws_f1
    p.config(compile="vitis", mode="sw_sim", project="Baseline")
    s = hcl.create_schedule([img, conv_w, dense_w], top)  

    # Create systolic array for Conv and Dense layers
    s[top.Conv_L1].systolic()
    s[top.Dense_L2].systolic()

    # Stream first SA's output to relu and flatten
    s.to(top.Conv_L1.L1, top.relu, depth=32)
    s.to(top.Flatten.FL, top.Dense_L2, depth=32)

    # Offload the main body to FPGA
    s.to([img, conv_w, dense_w], p.xcel)
    s.to(top.Dense_L2.L2, p.host)

    if stream:
        print("Stream SA output to ReLU module")
        s.to(top.MM_C.C, top.output)
    print(hcl.lower(s))

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
    
    f.inspect(args)

if __name__ == "__main__":
    ConvNet()

    

