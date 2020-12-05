import heterocl as hcl
import numpy as np
from itertools import permutations
import os

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
    test_autosa_integration()
    test_static_variable()
    test_weight_stationary_sa()
