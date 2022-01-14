import heterocl as hcl
import os, sys
import numpy as np

def test_gemm(target=None):

    A = hcl.placeholder((32, 32), "A")
    B = hcl.placeholder((32, 32), "B")

    def gemm(A, B):
        k = hcl.reduce_axis(0, 32, "k")
        C = hcl.compute((32, 32), lambda i, j:
                hcl.sum(A[i, k] * B[k, j], axis=k), "C")
        return C

    target = None # hcl.platform.zc706
    s = hcl.create_schedule([A, B], gemm)
    f = hcl.build(s, target)
    print(f)

def sample_gemm(m=1024, n=1024, k=1024, dtype=hcl.Int(), target=None):
    matrix_1 = hcl.placeholder((m, k))
    matrix_2 = hcl.placeholder((k, n))

    def kernel(matrix_1, matrix_2):
        r = hcl.reduce_axis(0, k, 'k')
        return hcl.compute((m, n),
                lambda y, x: hcl.sum(matrix_1[y, r] * matrix_2[r, x],
                                     axis=r),
                dtype=dtype,
                name="out_matrix")

    s = hcl.create_schedule([matrix_1, matrix_2], kernel)
    out_matrix = kernel.out_matrix
    block_size = 8
    y0, y1 = s[out_matrix].split(out_matrix.axis[0], factor=block_size)
    x0, x1 = s[out_matrix].split(out_matrix.axis[1], factor=block_size)
    s[out_matrix].reorder(y0, x0, y1, x1)

    f = hcl.build(s, target=target)
    print(f)

if __name__ == "__main__":
    test_gemm()
    sample_gemm()