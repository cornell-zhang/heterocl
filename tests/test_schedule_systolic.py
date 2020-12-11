import heterocl as hcl
import numpy as np
from itertools import permutations
import os

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
    s = hcl.create_schedule([W, X], kernel)
    pes = s.parallel(kernel.Y, axis=kernel.Y.axis[1])
    pe1, pe2, pe3 = pes

    # Data movement and broadcasting
    s.to(W, target.xcel)
    s.to(X, target.xcel).to([pe1, pe2, pe3])
    s.to(kernel.Y, target.host)

    # if there is no data movement information specified
    # then each undefined variable creates a port
    code = str(hcl.lower(s))
    assert "def Y_pe_3" in code, code
    assert "def Y_pe_2" in code, code
    assert "def Y_pe_1" in code, code

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

if __name__ == '__main__':
    test_stencil_stream()
    #test_autosa_integration()
    #test_static_variable()
    #test_weight_stationary_sa()
