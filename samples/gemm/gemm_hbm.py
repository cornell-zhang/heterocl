import heterocl as hcl
import numpy as np
import time
from gemm_main import gemm

def gemm_hbm(m=1024, n=1024, k=1024, dtype=hcl.UInt(32)):
    matrix_1 = hcl.placeholder((m, k), dtype=dtype, name="matrix_1")
    matrix_2 = hcl.placeholder((k, n), dtype=dtype, name="matrix_2")

    def kernel(matrix_1, matrix_2):
        r = hcl.reduce_axis(0, k, 'k')
        out_matrix = hcl.compute((m, n),
                lambda x, y: hcl.sum(matrix_1[x, r] * matrix_2[r, y],
                                     axis=r, dtype=dtype), dtype=dtype,
                name="out_matrix")
        return out_matrix

    s = hcl.create_schedule([matrix_1, matrix_2], kernel)

    config = {
        "host" : hcl.dev.cpu("intel", "e5"),
        "xcel" : [
            hcl.dev.fpga("xilinx", "xcvu19p")
        ]
    }
    target = hcl.platform.custom(config)
    target.config(compile="vitis", mode="hw_exe", backend="vhls")

    # block tiling and reorder
    out_matrix = kernel.out_matrix
    block_size = 4
    y0, y1 = s[out_matrix].split(out_matrix.axis[0], factor=block_size)
    x0, x1 = s[out_matrix].split(out_matrix.axis[1], factor=block_size)
    s[out_matrix].reorder(y0, x0, y1, x1)
    s[out_matrix].unroll(out_matrix.axis[1])
    s[out_matrix].unroll(out_matrix.axis[0])

    s.to(matrix_1, target.xcel.hbm[0])
    s.to(matrix_2, target.xcel.hbm[1])
    s.to(kernel.out_matrix, target.host.hbm[2])

    f = hcl.build(s, target=target)

    np_1 = np.random.randint(10, size=(m, k))
    np_2 = np.random.randint(10, size=(k, n))
    np_3 = np.matmul(np_1, np_2)

    hcl_m1 = hcl.asarray(np_1, dtype=dtype)
    hcl_m2 = hcl.asarray(np_2, dtype=dtype)
    hcl_m3 = hcl.asarray(np.zeros((m, n)), dtype=dtype)

    f(hcl_m1, hcl_m2, hcl_m3)
    print(hcl_m3.asnumpy())

def time_gemm(dtype, m=1024, n=1024, k=1024, target=None):
    hcl.init(dtype)
    f = gemm(m, n, k, dtype, target)
    np_1 = np.random.randint(10, size=(m, k))
    np_2 = np.random.randint(10, size=(k, n))
    np_3 = np.matmul(np_1, np_2)

    hcl_m1 = hcl.asarray(np_1, dtype=dtype)
    hcl_m2 = hcl.asarray(np_2, dtype=dtype)
    hcl_m3 = hcl.asarray(np.zeros((m, n)), dtype=dtype)
    f(hcl_m1, hcl_m2, hcl_m3)
    begin = time.time()
    for i in range(10):
        f(hcl_m1, hcl_m2, hcl_m3)
    end = time.time()
    print("dtype is: ", dtype)
    print("average of 10 runs takes: {} sec".format((end - begin) / 10))
    np.testing.assert_allclose(hcl_m3.asnumpy(), np_3, rtol=1e-03)

###############################################################################
# Test the algorithm with different data types
dtypes = [hcl.Int(32), hcl.Float(), hcl.Fixed(32, 16)]

# for dtype in dtypes:
# time_gemm(hcl.Float(), 10, 10, 10, 'sdaccel')
gemm_hbm()
time_gemm()
