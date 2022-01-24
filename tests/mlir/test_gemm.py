import heterocl as hcl
import os, sys
import numpy as np
import time

hcl.init(hcl.Float())

def test_gemm_cpu(target=None):

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
    A = np.random.randint(10, size=(32, 32)).astype(np.float32)
    B = np.random.randint(10, size=(32, 32)).astype(np.float32)
    C = np.zeros((32, 32), dtype=np.float32)
    f(A, B, C)
    golden = np.matmul(A, B)
    if (np.allclose(golden, C)):
        print("test_gemm passed")
    else:
        print("test_gemm failed")

def test_gemm_fpga(m=32, n=32, k=32, dtype=hcl.Int(), target=None):
    matrix_1 = hcl.placeholder((m, k), dtype=dtype, name="matrix1")
    matrix_2 = hcl.placeholder((k, n), dtype=dtype, name="matrix2")

    def kernel(matrix_1, matrix_2):
        r = hcl.reduce_axis(0, k, 'k')
        return hcl.compute((m, n),
                lambda y, x: hcl.sum(matrix_1[y, r] * matrix_2[r, x],
                                     axis=r, dtype=dtype),
                dtype=dtype,
                name="out_matrix")

    start_time = time.time()
    s = hcl.create_schedule([matrix_1, matrix_2], kernel)
    out_matrix = kernel.out_matrix
    block_size = 8
    y0, y1 = s[out_matrix].split(out_matrix.axis[0], factor=block_size)
    x0, x1 = s[out_matrix].split(out_matrix.axis[1], factor=block_size)
    s[out_matrix].reorder(y0, x0, y1, x1)

    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="csyn", project="gemm.prj")

    mod = hcl.build(s, target=target)
    end_time = time.time()
    print("Compilation time: {:.4f}ms".format((end_time-start_time)*1000))
    print(mod.src)
    mod()
    report = mod.report()
    report.display()

if __name__ == "__main__":
    # test_gemm_cpu()
    test_gemm_fpga()