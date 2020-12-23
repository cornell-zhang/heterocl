import heterocl as hcl
import numpy as np
from itertools import permutations
import os
import sys

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

    dtype = hcl.Float()
    input_image = hcl.placeholder((*shape, 3), name="input", dtype=dtype)
    s = hcl.create_schedule([input_image], jacobi)
    s[jacobi.output].stencil(unroll_factor=8)

    # Stream from grayscale to stencil module
    s.to(jacobi.gray, jacobi.output, depth=10)

    print(hcl.build(s, target='soda'))
    print(hcl.build(s, target='soda_xhls'))
    print(hcl.build(s, target='vhls'))


def test_autosa_integration():
    m=1024
    n=1024
    k=1024
    dtype=hcl.Int()

    matrix_1 = hcl.placeholder((m, k), dtype=dtype)
    matrix_2 = hcl.placeholder((k, n), dtype=dtype)

    def kernel(matrix_1, matrix_2):
        r = hcl.reduce_axis(0, k, 'k')
        return hcl.compute((m, n),
                lambda x, y: hcl.sum(matrix_1[x, r] * matrix_2[r, y],
                                     axis=r, dtype=dtype),
                dtype=dtype,
                name="out_matrix")

    s = hcl.create_schedule([matrix_1, matrix_2], kernel)
    out_matrix = kernel.out_matrix
    s[out_matrix].systolic()

    target = hcl.platform.aws_f1
    target.config(compile="vitis", mode="debug", backend="vhls")
    target.project = "test-autosa"
    f = hcl.build(s, target=target)
    print(f)


def test_static_variable():
    hcl.init()
    W = hcl.placeholder((3,), "W")
    X = hcl.placeholder((32,), "X")

    def kernel(W, X):
        k = hcl.reduce_axis(0, 3, "k")
        return hcl.compute((30,), lambda x: hcl.sum(X[x+k]*W[k], axis=k), "Y")
    
    target = hcl.platform.aws_f1
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
    assert "def Y_pe_3" in code, code
    assert "def Y_pe_2" in code, code
    assert "def Y_pe_1" in code, code
    print(code)

def test_weight_stationary_sa():
    hcl.init()
    W = hcl.placeholder((3,), "W")
    X = hcl.placeholder((32,), "X")

    def kernel(W, X):
        k = hcl.reduce_axis(0, 3, "k")
        return hcl.compute((30,), lambda x: 
            hcl.sum(X[x+k]*W[k], axis=k, name="sum"), "Y")
    
    config = {
        "host" : hcl.dev.cpu("intel", "e5"),
        "xcel" : [
             hcl.dev.fpga("xilinx", "xcvu19p")
        ]
    }
    p = hcl.platform.custom(config)
    p.config(compile="vitis", mode="debug", backend="vhls")
    s = hcl.create_schedule([W, X], kernel)

    pes = s.parallel(kernel.Y, axis=kernel.Y.axis[1])
    pe1, pe2, pe3 = pes

    # Data movement and broadcasting
    s.to(W, p.xcel)
    s.to(X, p.xcel).to([pe1, pe2, pe3])
    s.to(kernel.Y, p.host)

    # Move data betwen PEs
    # tensor "sum" is consumed by all the pe stages
    # we need differentiate from the regular tensor multi-casting
    s.to(pe1.sum, pe2).to(pe3).to(kernel.Y)
    print(hcl.lower(s))
    code = str(hcl.build(s, p))
    print(code)

# Simple read and write kernel
def test_inter_module_stream():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")

    def kernel(A):
        B = hcl.compute((10, 32), lambda *args: 0, "B")
        C = hcl.compute((10, 32), lambda *args: 0, "C")
        
        @hcl.def_([(10, 32), (10, 32)])
        def add(A, B):
            hcl.update(B, lambda *args: A[args] + 1)

        @hcl.def_([(10, 32), (10, 32)])
        def mul(B, C):
            hcl.update(C, lambda *args: B[args] * 2)
            
        add(A, B)
        mul(B, C)

    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A], kernel)
    
    # Stream one kernel's output to another's input
    s.to(kernel.add.B, kernel.mul.B)
    print(hcl.lower(s))

# GEMM example unrolling on two dimension
def test_2d_pe_unroll():
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
    # cannot easily handle unrolling imperfect loop

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
    pes = s.parallel(kernel.Y, axis=kernel.Y.axis[1])
    row1, row2 = pes

    # pe1, pe2, pe3 = pes
    print(hcl.lower(s))

if __name__ == '__main__':
    test_2d_pe_unroll()
    test_static_variable()    
    test_inter_module_stream()
    test_stencil_stream()
    test_autosa_integration()
    test_static_variable()
    test_weight_stationary_sa()
