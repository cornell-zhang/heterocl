import heterocl as hcl
import numpy as np
from itertools import permutations
import os
import sys

# Simplified interface for systolic array generation
def test_autosa_basic():
    m=64
    n=64
    k=64
    dtype=hcl.Int()

    A = hcl.placeholder((m,k), dtype=dtype, name="A")
    B = hcl.placeholder((k,n), dtype=dtype, name="B")

    def kernel(A, B):
        Y = hcl.compute((m, n), lambda *args: 0, name="Y0")
        with hcl.Stage("Y"):
            with hcl.for_(0, m) as i:
                with hcl.for_(0, n) as j:
                    Y[i][j] = 0
                    with hcl.for_(0, k) as r:
                        Y[i][j] += A[i][r] * B[r][j]

    p = hcl.platform.aws_f1
    p.config(compile="vitis", mode="debug")
    s = hcl.create_schedule([A, B], kernel)

    # Apply tranpose and packing automatically
    s[kernel.Y].systolic()
    print(hcl.lower(s))


# Make AutoSA aware of the input data layout and 
def test_autosa_pack():
    m=64
    n=64
    k=64
    # Systolic array size
    sa_dim_x = 4
    sa_dim_y = 4
    dtype=hcl.Int()

    A = hcl.placeholder((m,k), dtype=dtype, name="A")
    B = hcl.placeholder((k,n), dtype=dtype, name="B")

    def kernel(A, B):
        Y = hcl.compute((m, n), lambda *args: 0, name="Y0")
        with hcl.Stage("Y"):
            with hcl.for_(0, m) as i:
                with hcl.for_(0, n) as j:
                    Y[i][j] = 0
                    with hcl.for_(0, k) as r:
                        Y[i][j] += A[i][r] * B[r][j]

    p = hcl.platform.aws_f1
    p.config(compile="vitis", mode="debug")
    s = hcl.create_schedule([A, B], kernel)

    # Both A and B are packed into 512 bit 
    # and streamed into systolic array
    # data unpacking handled automatically
    s.transpose(B).pack(factor=512).to(p.xcel).to(kernel.Y)
    s.pack(A, factor=512).to(p.xcel).to(kernel.Y)
    
    # Array partitioning (specify SA size)
    i, j = kernel.Y.axis[0], kernel.Y.axis[1]
    i_outer, j_outer, i_inner, j_inner = s[kernel.Y].tile(i, j, sa_dim_x, sa_dim_y)

    # Simple case: output stationary 2D SA (space loop: i, j)
    PEs = s.parallel(kernel.Y, axis=[i_inner, j_inner])
    assert PEs.size == (4,4)

    # Data movement horizontally
    [ s.to(PEs[k,0].A, PEs[k,1])
        .to(PEs[k,2]).to(PEs[k,3]) for k in range(sa_dim_x) ]

    # Data movement vertically
    [ s.to(PEs[0,k].B, PEs[1,k])
        .to(PEs[2,k]).to(PEs[3,k]) for k in range(sa_dim_y) ]

    # Drain output to host memory
    [ s.to(PEs[i,j].Y0, kernel.Y.Y0)
        .to(p.host) for i in range(sa_dim_x) 
            for j in range(sa_dim_y) ]

    print(hcl.lower(s))

if __name__ == '__main__':
    test_autosa_basic()
    # test_autosa_pack()