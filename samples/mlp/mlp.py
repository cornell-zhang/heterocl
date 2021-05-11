import heterocl as hcl
import numpy as np

def matmul(A, B, name="Y0"):
    assert B.shape[0] == A.shape[1]
    m, k = A.shape
    _, n = B.shape
    Y = hcl.compute((m, n), lambda *args: 0, dtype=A.dtype, name=name)
    with hcl.Stage(f"MM_{name}"):
        with hcl.for_(0, m, name="i") as i:
            with hcl.for_(0, n, name="j") as j:
                Y[i][j] = 0
                with hcl.for_(0, k, name="k") as r:
                    Y[i][j] += A[i][r] * B[r][j]   
    return Y 

def flatten(op, name="FL"):
    new_shape = (1, np.prod(op.shape))
    new_tensor = hcl.compute(new_shape, lambda *args: 0, name=name)
    with hcl.Stage(f"Flatten"):
        with hcl.for_(0, op.shape[0]) as y:
            with hcl.for_(0, op.shape[1]) as x:
                new_tensor[0][y * op.shape[1] + x] = op[y,x]
    return new_tensor

def relu(op, name="relu"):
    @hcl.def_([()])
    def select(A):
        temp = hcl.scalar(A)
        hcl.return_(hcl.select(temp > 0.0, temp, 0.0))
    return hcl.compute(op.shape, 
        lambda *args: select(op[args]), name=name)

def MLP(stream=False):
    m=8
    n=8
    k=8
    dtype=hcl.Float()
    hcl.init(dtype)

    img = hcl.placeholder((m, k), dtype=dtype, name="input_image")
    w1  = hcl.placeholder((k, n), dtype=dtype, name="w1")  
    w2  = hcl.placeholder((m*n, 10), dtype=dtype, name="w2")

    # A two layer MLP example 
    def top(opA, opB, opC):
        C = matmul(opA, opB, name="L1")
        D = flatten(relu(C, name="relu")) # D is the flattened tensor (1,64)
        return matmul(D, opC, name="L2")  # return onehot pred (1,10)

    p = hcl.Platform.aws_f1
    p.config(compile="vitis", mode="sw_sim", project="MLP")
    s = hcl.create_schedule([img, w1, w2], top)  

    # Create systolic array for two MLP layers
    # 1. Update the codegen rule about data serialization/pack (i.e. how the input
    #    data is packed and serialized/deserialized)
    # 2. 
    s[top.MM_L1].systolic()
    s[top.MM_L2].systolic()

    # Stream relu's output to Flatten layer
    s.to(top.relu, top.Flatten, depth=32)

    # Stream first SA's output to relu and flatten
    # s.to(top.MM_L1.L1, top.relu, depth=32)
    # s.to(top.Flatten.FL, top.MM_L2, depth=32)

    # Offload the main body to FPGA
    s.to([img, w1, w2], p.xcel)
    s.to(top.MM_L2.L2, p.host)

    if stream:
        print("Stream SA output to ReLU module")
        s.to(top.MM_C.C, top.output)
    print(hcl.lower(s))

    f = hcl.build(s, target=p)
    args = list()
    low, high = 0, 10
    args.append(np.random.uniform(low=low, high=high, size=img.shape))
    args.append(np.random.uniform(low=low, high=high, size=w1.shape))
    args.append(np.random.uniform(low=low, high=high, size=w2.shape))
    args.append(np.random.uniform(low=low, high=high, size=(10,)))
    f.inspect(args)

if __name__ == "__main__":
    MLP()

    
