import heterocl as hcl
import numpy as np
import time

def gemm_vhls(m=1024, n=1024, k=1024, dtype=hcl.Int()):
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

    target = hcl.Platform.xilinx_zc706
    s.to([matrix_1, matrix_2], target.xcel, local_buffer=False)
    s.to(kernel.out_matrix, target.host, local_buffer=False)
    target.config(compile="vivado_hls", mode="csim|csyn")

    f = hcl.build(s, target=target)
    return f

def time_gemm(dtype, m=1024, n=1024, k=1024):
    hcl.init(dtype)
    f = gemm_vhls(m, n, k, dtype)
    np_1 = np.random.randint(10, size=(m, k))
    np_2 = np.random.randint(10, size=(k, n))
    np_3 = np.matmul(np_1, np_2)

    hcl_m1 = hcl.asarray(np_1, dtype=dtype)
    hcl_m2 = hcl.asarray(np_2, dtype=dtype)
    hcl_m3 = hcl.asarray(np.zeros((m, n)), dtype=dtype)
    begin = time.time()
    f(hcl_m1, hcl_m2, hcl_m3)
    end = time.time()
    print("dtype is: ", dtype)
    print("simulation time: {} sec".format((end - begin)))

# test very large gemm simulation
time_gemm(hcl.Int(32), 1024, 1024, 1024)

dtypes = [hcl.Int(32), hcl.Float(), hcl.Fixed(32, 16)]
for dtype in dtypes:
    time_gemm(hcl.Int(32), 10, 10, 10)

