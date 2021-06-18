import heterocl as hcl
import numpy as np
from itertools import permutations
import os
import sys

def matmul(A, B, name="Y0"):
    assert B.shape[0] == A.shape[1]
    m, k = A.shape
    _, n = B.shape
    Y = hcl.compute((m, n), lambda *args: 0, dtype=A.dtype, name=name)
    with hcl.Stage(f"MM_{name}"):
        with hcl.for_(0, m) as i:
            with hcl.for_(0, n) as j:
                Y[i][j] = 0
                with hcl.for_(0, k) as r:
                    Y[i][j] += A[i][r] * B[r][j]   
    return Y 

def relu(op, name="C"):
    @hcl.def_([()])
    def select(A):
        temp = hcl.scalar(A)
        hcl.return_(hcl.select(temp > 0.0, temp, 0.0))
    return hcl.compute(op.shape, 
        lambda *args: select(op[args]), name=name)

def test_inter_systolic_array_conn():
    m=64
    n=64
    k=64
    dtype=hcl.Float()
    hcl.init(dtype)
    A = hcl.placeholder((m, k), dtype=dtype, name="A")
    B = hcl.placeholder((k, n), dtype=dtype, name="B") 
    C = hcl.placeholder((n, n), dtype=dtype, name="C")

    def top(opA, opB, opC):
        K = matmul(opA, opB, "SA1")
        return matmul(K, opC, "SA2")

    p = hcl.Platform.aws_f1
    p.config(compile="vitis", mode="sw_sim", project="s1-autosa")
    s = hcl.create_schedule([A, B, C], top)  

    # Create two systolic arrays
    s[top.MM_SA1].systolic()
    s[top.MM_SA2].systolic()

    # Move inputs and outputs
    s.to([A, B, C], p.xcel)
    s.to(top.MM_SA2.SA2, p.host)

    print(hcl.lower(s))
    f = hcl.build(s, target=p)

    args = list()
    low, high = 0, 10
    args.append(np.random.uniform(low=low, high=high, size=A.shape))
    args.append(np.random.uniform(low=low, high=high, size=B.shape))
    args.append(np.random.uniform(low=low, high=high, size=C.shape))
    args.append(np.random.uniform(low=low, high=high, size=(m,n)))
    f.inspect(args)

# A simple test case for connecting SA and other mods
def test_compose_systolic_arrays(stream=False):
    m=64
    n=64
    k=64
    dtype=hcl.Float()
    hcl.init(dtype)
    A = hcl.placeholder((m, k), dtype=dtype, name="A")
    B = hcl.placeholder((k, n), dtype=dtype, name="B")  

    # A single layer MLP example 
    def top(opA, opB):
        C = matmul(opA, opB, "C")
        return relu(C, "output")

    p = hcl.Platform.aws_f1
    p.config(compile="vitis", mode="sw_sim", project="s1-autosa")
    s = hcl.create_schedule([A, B], top)  

    s[top.MM_C].systolic()
    s.to([A, B], p.xcel)
    s.to(top.output, p.host)
    if stream:
        print("Stream SA output to ReLU module")
        s.to(top.MM_C.C, top.output)
    print(hcl.lower(s))

    f = hcl.build(s, target=p)
    args = list()
    low, high = 0, 10
    args.append(np.random.uniform(low=low, high=high, size=A.shape))
    args.append(np.random.uniform(low=low, high=high, size=B.shape))
    args.append(np.random.uniform(low=low, high=high, size=(m,n)))
    f.inspect(args)

# A dummy free-running kernel. 
def test_free_running_kernel():
    length = 10
    hcl.init()
    dtype=hcl.Float()

    op1 = hcl.placeholder((length, ), dtype=dtype, name="op1")
    op2 = hcl.placeholder((length, ), dtype=dtype, name="op2")
    out = hcl.placeholder((length, ), dtype=dtype, name="out")

    def top(A, B, C):
        index = hcl.scalar(0)
        with hcl.while_(1):
            C[index.v] = A[index.v] + B[index.v]
            index.v += 1

    p = hcl.Platform.aws_f1
    p.config(compile="vitis", mode="debug")
    s = hcl.create_schedule([op1, op2, out], top)

    s.to([op1, op2], p.xcel, mode=hcl.IO.Stream)
    s.to(out, p.host, mode=hcl.IO.Stream)
    ir = str(hcl.lower(s))

    assert "op1[0].read"  in ir
    assert "op2[0].read"  in ir
    assert "out[0].write" in ir


def test_autosa_schedule():
    m=3
    n=3
    k=3
    dtype=hcl.Int()

    matrix_1 = hcl.placeholder((m, k), dtype=dtype, name="W")
    matrix_2 = hcl.placeholder((k, n), dtype=dtype, name="X")

    def kernel(matrix_1, matrix_2):
        # imperative (without transposed matrix B)
        Y = hcl.compute((m, n), lambda *args: 0, name="Y0")
        with hcl.Stage("Y"):
            with hcl.for_(0, m) as i:
                with hcl.for_(0, n) as j:
                    Y[i][j] = 0
                    with hcl.for_(0, k) as r:
                        Y[i][j] += matrix_1[i][r] * matrix_2[r][j]

        # r = hcl.reduce_axis(0, k, 'k')
        # return hcl.compute((m, n), lambda x, y: 
        #     hcl.sum(matrix_1[x, r] * matrix_2[r, y], axis=r, dtype=dtype),
        #         dtype=dtype, name="Y")

    p = hcl.Platform.aws_f1
    p.config(compile="vitis", mode="debug")
    s = hcl.create_schedule([matrix_1, matrix_2], kernel)

    # The unrolled PEs scheduling is handled by AutoSA 
    PEs = s.parallel(kernel.Y, axis=[0,1])
    assert PEs.size == (3,3)
    print(PEs[0,0].op.inputs)
    print(PEs[:,0])

    # Schedule with .to using high order function
    # Example of 3x3 inputs braodcasting and inter-PE streaming
    # 1. Broadcasting matrix X from left
    # 2. Broadcasting matrix W from top
    s.to(matrix_1, p.xcel).to(PEs[0,:], mode=hcl.IO.Stream)
    s.to(matrix_2, p.xcel).to(PEs[:,0], mode=hcl.IO.DMA)

    # 3. Create inter-PE streaming channels
    higher_order_func = False
    if not higher_order_func:
        [ s.to(PEs[k,0].X, PEs[k,1]).to(PEs[k,2]) for k in range(m) ]
        [ s.to(PEs[0,k].W, PEs[1,k]).to(PEs[2,k]) for k in range(n) ]
        [ s.to(PEs[i,j].Y0, kernel.Y.Y0).to(p.host) for i in range(m) for j in range(n) ]
    # Or in other ways
    else:
        map(lambda k: s.to(PEs[k,0].X, PEs[k,1]).to(PEs[k,2]), range(m))
        map(lambda k: s.to(PEs[0,k].W, PEs[1,k]).to(PEs[2,k]), range(n))
        map(lambda i,j: s.to(PEs[i,j], kernel.Y.Y0).to(p.host), range(m), range(n))

    # Check the annotation embedded in th IR
    print(hcl.lower(s))


# Parse the generated by AutoSA
def test_autosa_gemm():
    m=64
    n=64
    k=64
    dtype=hcl.Int()

    matrix_1 = hcl.placeholder((m, k), dtype=dtype, name="W")
    matrix_2 = hcl.placeholder((k, n), dtype=dtype, name="X")

    def kernel(matrix_1, matrix_2):
        # imperative (without transposed matrix B)
        Y = hcl.compute((m, n), lambda *args: 0, name="Y0")
        with hcl.Stage("Y"):
            with hcl.for_(0, m) as i:
                with hcl.for_(0, n) as j:
                    Y[i][j] = 0
                    with hcl.for_(0, k) as r:
                        Y[i][j] += matrix_1[i][r] * matrix_2[r][j]

        # r = hcl.reduce_axis(0, k, 'k')
        # return hcl.compute((m, n), lambda x, y: 
        #     hcl.sum(matrix_1[x, r] * matrix_2[r, y], axis=r, dtype=dtype),
        #         dtype=dtype, name="Y")

    target = hcl.Platform.aws_f1
    target.config(compile="vitis", mode="debug")
    s = hcl.create_schedule([matrix_1, matrix_2], kernel)

    # Do extern ip
    s[kernel.Y].systolic()
    print(hcl.build(s, target))

def test_stencil_stream():
    shape = (480, 640)
    def jacobi(input_image):
        tensor = hcl.compute(shape, lambda y, x: 
            input_image[y,x,0]*0.3 + input_image[y,x,1]*0.59+input_image[y,x,2]*0.11 , "gray")
        def jacobi_kernel(y, x):
            return (tensor[y+1, x-1] +
                    tensor[y  , x  ] +
                    tensor[y+1, x  ] +
                    tensor[y+1, x+1] +
                    tensor[y+2, x  ]) / 5

        return hcl.compute(shape, jacobi_kernel, name="output")

    hcl.init()
    dtype = hcl.Float()
    input_image = hcl.placeholder((*shape, 3), name="input", dtype=dtype)
    p = hcl.Platform.aws_f1
    p.config(compile="vitis", mode="sw_sim")

    s = hcl.create_schedule([input_image], jacobi)
    s[jacobi.output].stencil(unroll_factor=8)

    # Create FIFO channels
    s.to(input_image, p.xcel)
    s.to(jacobi.gray, jacobi.output, depth=10)
    s.to(jacobi.output, p.host, mode=hcl.IO.Stream)

    code = str(hcl.build(s, target='soda'))
    code = str(hcl.build(s, target='soda_xhls'))
    code = str(hcl.build(s, target='vhls'))

    args = hcl.util.gen_hcl_array(s)
    f = hcl.build(s, p)
    f.inspect(args)

def test_static_variable():
    hcl.init()
    W = hcl.placeholder((3,), "W")
    X = hcl.placeholder((32,), "X")

    def kernel(W, X):
        k = hcl.reduce_axis(0, 3, "k")
        return hcl.compute((30,), lambda x: hcl.sum(X[x+k]*W[k], axis=k), "Y")
    
    target = hcl.Platform.aws_f1
    sch = hcl.create_schedule([W, X], kernel)

    # The loop reorder seems not to work on imperfect loops
    # sch[Y].reorder(Y.axis[0], Y.axis[1])

    # The analysis is offloaded to AutoSA and we just inject 
    # information into the IR
    pes = sch.parallel(kernel.Y, axis=kernel.Y.axis[1])
    pe1, pe2, pe3 = pes

    # Data movement and broadcasting
    sch.to(W, target.xcel)
    sch.to(X, target.xcel).to([pe1, pe2, pe3])
    sch.to(kernel.Y, target.host)

    # if there is no data movement information specified
    # then each undefined variable creates a port
    code = str(hcl.lower(sch))
    assert "Y_pe_0" in code, code
    assert "Y_pe_1" in code, code
    assert "Y_pe_2" in code, code

def test_weight_stationary_sa():
    hcl.init()
    W = hcl.placeholder((3,), "W")
    X = hcl.placeholder((32,), "X")
    Y = hcl.placeholder((30,), "Y")

    """
    # Original program
    for (int i = 0, 30) 
     for (int k = 0, 3) 
       y[i] += w[k] * x[i+k]

    # Unrolled version
    # Analyze the movement between unrolled PEs
    for (int i = 0, 30) {
       y[i] += w[0] * x[i]
       y[i] += w[1] * x[i+1]
       y[i] += w[2] * x[i+2]
    }

    # Expected program after mutation
    void top() {

      // initialize weights
      w = memcpy(w)

      #pragma HLS dataflow
      // data loader
      for (int i = 0, 32) {
          int temp = x[i];
          x1 = temp;
          x2 = temp;
          x3 = temp;

          // initialization
          y0 = 0;
      }

      // same function
      pe1(x1, w[0], y0, y1);
      pe2(x2, w[1], y1, y2);
      pe3(x3, w[2], y2, y3);

      // data drainer
      for (int i = 0, 32) {
        int out = y3;  
        if (i > 2) {
          y[i-2] = out; 
        }
      }
    }
    """

    def kernel(W, X, Y):
        with hcl.Stage("conv1d") as stage:
            with hcl.for_(0, 30, name="i") as i:
                Y[i] = 0
                with hcl.for_(0, 3, name="k") as k:
                    Y[i] += W[k] * X[k+i]
    
    config = {
        "host" : hcl.dev.cpu("intel", "e5"),
        "xcel" : [
             hcl.dev.fpga("xilinx", "xcvu19p")
        ]
    }

    p = hcl.Platform.custom(config)
    p.config(compile="vitis", mode="debug", backend="vhls")
    s = hcl.create_schedule([W, X, Y], kernel)

    # unrolling reduction loop k
    #  x3,x2,x1 ---->------>------->
    #             ----    ----    ----
    #  y3,y2,y1 -> w1      w2      w3
    #             ----    ----    ----
    #  y1 = w1x1 + w2x2 + w3x3
    #  y2 = w1x2 + w2x3 + w3x4

    print("[ INFO ] space loop ", kernel.conv1d.axis[1]) 
    pes = s.parallel(kernel.conv1d, axis=kernel.conv1d.axis[1])

    # Each PE is an individual module
    pe1, pe2, pe3 = pes

    # Data movement 
    # 1. W moved into FPGA's on-chip buffer
    s.to(W, p.xcel, burst=True)

    # 2. Broadcast x[i] into PEs
    s.to(X, p.xcel).to([pe1, pe2, pe3])

    # 3. Move data (partial sum) betwen PEs and finally to host
    s.to(pe1.Y, pe2).to(pe3).to(p.host)
    print(hcl.lower(s))


# GEMM example unrolling on two dimension
def test_two_loops():
    m=2
    n=2
    k=2
    dtype=hcl.Int()

    matrix_1 = hcl.placeholder((m, k), dtype=dtype, name="W")
    matrix_2 = hcl.placeholder((k, n), dtype=dtype, name="X")

    def kernel(matrix_1, matrix_2):
        r = hcl.reduce_axis(0, k, 'k')
        return hcl.compute((m, n), lambda x, y: 
            hcl.sum(matrix_1[x, r] * matrix_2[r, y], axis=r, dtype=dtype),
                dtype=dtype, name="Y")

    s = hcl.create_schedule([matrix_1, matrix_2], kernel)
    # Example body of s[kernel.Y].op:
    # for "stage_name"="Y" (x, 0, 2) {
    #   // attr [iter_var(x, Range(min=0, extent=2))] loop_scope = x
    #   for "stage_name"="Y" (y, 0, 2) {
    #     // attr [iter_var(y, Range(min=0, extent=2))] loop_scope = y
    #     // attr [buffer(sum, 0x55e801b07bd0)] attach_scope = "Y"
    #     for "stage_name"="Y" (k, 0, 2) {
    #       // attr [iter_var(k, Range(min=0, extent=2))] loop_scope = k
    #       sum[0] = W[(k + (x*2))])*X[(y + (k*2))] + sum[0]
    #     }
    #     Y[(y + (x*2))] = int32(sum[0])
    #   }
    # }

    # Unroll the two innermost loops to PE array
    # Has to be in-order (from outermost to innermost)
    axes = [ kernel.Y.axis[1], kernel.Y.axis[2] ]
    PEs = s.parallel(kernel.Y, axis=axes)

    # The stage layout should be 
    # 1. Multiple substages attaching to the parent stage
    # 2. The parent stage includes the original body and attaching anchors
    print(PEs[0][0].op)
    print(PEs[0][1].op)
    print(PEs[1][0].op)
    print(PEs[1][1].op)

    # Each PE body is marked as virtual stage
    # we will keep the original body for actual code generation
    # The layout information is embedded using .to()
    print(hcl.lower(s))

def test_unroll_outer_loops():
    m=2
    n=2
    k=2
    dtype=hcl.Int()

    matrix_1 = hcl.placeholder((m, k), dtype=dtype, name="W")
    matrix_2 = hcl.placeholder((k, n), dtype=dtype, name="X")

    def kernel(matrix_1, matrix_2):
        r = hcl.reduce_axis(0, k, 'k')
        return hcl.compute((m, n), lambda x, y: 
            hcl.sum(matrix_1[x, r] * matrix_2[r, y], axis=r, dtype=dtype),
                dtype=dtype, name="Y")

    s = hcl.create_schedule([matrix_1, matrix_2], kernel)

    # unroll the spatial dimension 
    # keep the temporal dimension inside the PE
    axes = [ kernel.Y.axis[1], kernel.Y.axis[2] ]
    pes = s.parallel(kernel.Y, axis=kernel.Y.axis[1])
    pe1, pe2 = pes
    code = str(hcl.lower(s))

if __name__ == '__main__':
    test_weight_stationary_sa()

    test_two_loops() 
    test_stencil_stream()
    test_inter_systolic_array_conn()
    test_compose_systolic_arrays(True)
    test_free_running_kernel()
    
    test_compose_systolic_arrays()
    test_autosa_schedule()
    test_static_variable()
    
    test_autosa_gemm()
    test_unroll_outer_loops() 
    test_weight_stationary_sa()
