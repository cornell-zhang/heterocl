# Yang.Bai
# yb269@cornell.edu

import heterocl as hcl
import numpy as np

hcl.init()

# matrix_size = (16, 16)
# def add_compute(A, B):
#     C = hcl.compute(A.shape, lambda x, y: A[x, y] + B[x, y], "C")
#     return C

# def add_compute_2(A, B):
#     C = hcl.compute(A.shape, lambda x: A[x] + B[x], "C")
#     return C

# A = hcl.placeholder(matrix_size, "A")
# B = hcl.placeholder(matrix_size, "B")

# s = hcl.create_schedule([A, B], add_compute)
# # f2 = hcl.build(s, target='sdaccel')
# f2 = hcl.build(s, target='aocl')
# print (f2)

# hcl_A = hcl.asarray(np.random.random_sample(matrix_size), dtype = hcl.Float())
# hcl_B = hcl.asarray(np.random.random_sample(matrix_size), dtype = hcl.Float())
# hcl_C = hcl.asarray(np.zeros(matrix_size), dtype = hcl.Float())
# hcl_C2 = hcl.asarray(np.zeros(matrix_size), dtype = hcl.Float())
# f3 = hcl.build(s)

# A = hcl.placeholder((10, ), "A")
# B = hcl.placeholder((10, ), "B")
# s = hcl.create_schedule([A, B], add_compute_2)
# f4 = hcl.build(s, target='sdaccel')
# print (f4)
# print (hcl_A, hcl_B, hcl_C)

matrix_1_size = (10, 10)
matrix_2_size = (10, 10)
matrix_3_size = (matrix_1_size[0], matrix_2_size[1])

def gemm_compute(matrix_1, matrix_2):
    m = matrix_1.shape[0];
    k = matrix_1.shape[1];
    n = matrix_2.shape[1];
    r = hcl.reduce_axis(0, k, 'k')
    temp = hcl.compute((m, n), 
            lambda x, y: hcl.sum(matrix_1[x, r] * matrix_2[r, y], 
            axis = r), name='matrix_3')
    return temp

matrix_1 = hcl.placeholder(matrix_1_size)
matrix_2 = hcl.placeholder(matrix_2_size)

s = hcl.create_schedule([matrix_1, matrix_2], gemm_compute)
f = hcl.build(s, target='sdaccel_csim')
code = hcl.build(s, target='aocl')
with open('gemm_aocl.cl', 'w') as fin:
    fin.write(code)

code2 = hcl.build(s, target='sdaccel')
with open('gemm_sdaccel.cl', 'w') as fin2:
    fin2.write(code2)


matrix_1_np = np.random.randint(10, size=matrix_1_size)
matrix_2_np = np.random.randint(10, size=matrix_2_size)
matrix_3_np = np.random.randint(10, size=matrix_3_size)

hcl_matrix_1 = hcl.asarray(matrix_1_np)
hcl_matrix_2 = hcl.asarray(matrix_2_np)
hcl_matrix_3 = hcl.asarray(matrix_3_np)

# f(hcl_matrix_1, hcl_matrix_2, hcl_matrix_3)





# with open('sdaccel.cl', 'w') as f:
#     f.write(code)




