import heterocl as hcl
import numpy as np
from itertools import permutations
import os

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
    
    target = hcl.platform.aws_f1
    s = hcl.create_schedule([W, X], kernel)
    pes = s.parallel(kernel.Y, axis=kernel.Y.axis[1])
    pe1, pe2, pe3 = pes

    # Data movement and broadcasting
    s.to(W, target.xcel)
    s.to(X, target.xcel).to([pe1, pe2, pe3])
    s.to(kernel.Y, target.host)

    # Move data betwen PEs
    # tensor "sum" is consumed by all the pe stages
    # we need differentiate from the regular tensor multi-casting
    s.to(pe1.sum, pe2).to(pe3).to(kernel.Y)
    code = str(hcl.lower(s))
    print(code)

if __name__ == '__main__':
    test_static_variable()
    test_weight_stationary_sa()
