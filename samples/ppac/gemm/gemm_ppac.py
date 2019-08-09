"""
author: Guyue Huang (gh424@cornell.edu)

General Matrix Multiplication
target : rv64_ppac
"""
"""
modified on Aug 1
bit_width  = 1
m, n, k    = 16, 2, 64
"""

import heterocl as hcl
import numpy as np


def gemm(m, n, k, dtype=hcl.Int(), target=None):
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
    f = hcl.build(s, target=target, name='gemm')
    return f

dtype = hcl.UInt(8)
hcl.init(dtype)
m, n, k = 4, 4, 64
f = gemm(m, n, k, dtype, target="rv64_ppac")

print(f)
"""
with open("csrc.cc", "w") as ofile:
    ofile.write('/*CodeGenC backend*/\n'+str(f))
ofile.close()
"""