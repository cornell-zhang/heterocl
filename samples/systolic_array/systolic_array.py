import heterocl as hcl
import numpy as np
import time

m = k = n = 16
x_max = y_max = 16

def gemm(m=16, n=16, k=16, dtype=hcl.Int(), target=None):
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
    block_size = 8
    y0, y1 = s[out_matrix].split(out_matrix.axis[0], factor=block_size)
    x0, x1 = s[out_matrix].split(out_matrix.axis[1], factor=block_size)
    s[out_matrix].reorder(y0, x0, y1, x1)

    f = hcl.build(s, target=target)
    return f

def systolic(m=16, k=16, n=16, dtype=hcl.Int(), target=None):
    hcl.init(dtype)

    dim_x, dim_y = 16, 16
    m_A  = hcl.placeholder((m, k), dtype=dtype, name="m_A")
    m_B  = hcl.placeholder((k, n), dtype=dtype, name="m_B")
    m_output = hcl.placeholder((m, n), dtype=dtype, name="m_output")

    # k (time) and y/x (spatial) dim
    def kernel(k, y, x):
        last = hcl.scalar(
            hcl.select(k==0, 0, m_output[y, x]), "last")
        m_output[y, x] = last.v + m_A[y, k] * m_B[k, x] 

    hcl.mutate((m, dim_y, dim_x), 
        lambda k, y, x: kernel(k, y, x))
    s = hcl.create_schedule([m_A, m_B, m_output])
    f = hcl.build(s, target=target)
    return f
    
dtype = hcl.Int()
fg = gemm(m, n, k, dtype=hcl.Int(), target="llvm")
fs = systolic(m, n, k, dtype=hcl.Int(), target="llvm")

np_1 = np.random.randint(10, size=(m, k))
np_2 = np.random.randint(10, size=(k, n))
np_3 = np.matmul(np_1, np_2)

hcl_m1 = hcl.asarray(np_1, dtype=dtype)
hcl_m2 = hcl.asarray(np_2, dtype=dtype)
hcl_m3 = hcl.asarray(np.zeros((m, n)), dtype=dtype)
hcl_m4 = hcl.asarray(np.zeros((m, n)), dtype=dtype)

fg(hcl_m1, hcl_m2, hcl_m3)
fs(hcl_m1, hcl_m2, hcl_m4)
assert np.array_equal(hcl_m3.asnumpy(), hcl_m4.asnumpy())
