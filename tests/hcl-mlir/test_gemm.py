# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

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
        C = hcl.compute((32, 32), lambda i, j: hcl.sum(A[i, k] * B[k, j], axis=k), "C")
        return C

    target = None  # hcl.platform.zc706
    s = hcl.create_schedule([A, B], gemm)
    f = hcl.build(s, target)
    A = np.random.randint(10, size=(32, 32)).astype(np.float32)
    B = np.random.randint(10, size=(32, 32)).astype(np.float32)
    C = np.zeros((32, 32), dtype=np.float32)
    hcl_A = hcl.asarray(A)
    hcl_B = hcl.asarray(B)
    hcl_C = hcl.asarray(C)
    f(hcl_A, hcl_B, hcl_C)
    golden = np.matmul(A, B)
    if np.allclose(golden, hcl_C.asnumpy()):
        print("test_gemm passed")
    else:
        print("test_gemm failed")


def test_gemm_fpga(m=32, n=32, k=32, dtype=hcl.Int(), target=None):
    matrix_1 = hcl.placeholder((m, k), dtype=dtype, name="matrix1")
    matrix_2 = hcl.placeholder((k, n), dtype=dtype, name="matrix2")

    def kernel(matrix_1, matrix_2):
        r = hcl.reduce_axis(0, k, "k")
        return hcl.compute(
            (m, n),
            lambda y, x: hcl.sum(matrix_1[y, r] * matrix_2[r, x], axis=r, dtype=dtype),
            dtype=dtype,
            name="out_matrix",
        )

    start_time = time.time()
    s = hcl.create_schedule([matrix_1, matrix_2], kernel)
    out_matrix = kernel.out_matrix
    block_size = 8
    y0, y1 = s[out_matrix].split(out_matrix.axis[0], factor=block_size)
    x0, x1 = s[out_matrix].split(out_matrix.axis[1], factor=block_size)
    s[out_matrix].reorder(y0, x0, y1, x1)

    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="debug", project="gemm.prj")

    mod = hcl.build(s, target=target)
    end_time = time.time()
    print("Compilation time: {:.4f}ms".format((end_time - start_time) * 1000))
    print(mod.src)
    mod()
    report = mod.report()
    report.display()


def test_gemm_ihls(M=32, N=32, K=32, dtype=hcl.Int(), target=None):
    hcl.init(hcl.Int())
    A = hcl.placeholder((M, K), name="A")
    B = hcl.placeholder((K, N), name="B")

    def gemm(A, B):
        k = hcl.reduce_axis(0, K, name="k")
        C = hcl.compute((32, 32), lambda i, j: hcl.sum(A[i, k] * B[k, j], axis=k), "C")
        return C

    s = hcl.create_schedule([A, B], gemm)

    # optimization
    s_C = gemm.C
    i0, i1 = s[s_C].split(s_C.axis[0], 8)
    j0, j1 = s[s_C].split(s_C.axis[1], 8)
    s[s_C].reorder(i0, j0, i1, j1)
    s[s_C].unroll(j1)
    s[s_C].pipeline(i1)
    print(hcl.lower(s))

    f = hcl.build(s, "ihls")
    print(f)


def test_gemm_outline(M=32, N=32, K=32, dtype=hcl.Int(), target=None):
    hcl.init(hcl.Int())
    A = hcl.placeholder((M, K), name="A")
    B = hcl.placeholder((K, N), name="B")

    def gemm(A, B):
        k = hcl.reduce_axis(0, K, name="k")
        C = hcl.compute((32, 32), lambda i, j: hcl.sum(A[i, k] * B[k, j], axis=k), "C")
        return C

    s = hcl.create_schedule([A, B], gemm)

    # optimization
    s_C = gemm.C
    i0, i1 = s[s_C].split(s_C.axis[0], 8)
    j0, j1 = s[s_C].split(s_C.axis[1], 8)
    s[s_C].reorder(i0, j0, i1, j1)
    s[s_C].unroll(j1)
    s[s_C].pipeline(i1)
    # s.outline(s[s_C])
    print(hcl.lower(s))

    f = hcl.build(s, "vhls")
    print(f)


def test_gemm_buffer(M=32, N=32, K=32, dtype=hcl.Int(), target=None):
    hcl.init(hcl.Float())
    A = hcl.placeholder((M, K), name="A")
    B = hcl.placeholder((K, N), name="B")

    def gemm(A, B):
        k = hcl.reduce_axis(0, K, name="k")
        C = hcl.compute((M, N), lambda i, j: hcl.sum(A[i, k] * B[k, j], axis=k), "C")
        return C

    s = hcl.create_schedule([A, B], gemm)

    # optimization
    C = gemm.C
    # j_out, j_in = s[C].split(C.axis[1], 2)
    # s[C].reorder(j_in, C.axis[2])
    # s.buffer_at(C, s[C], j_out)
    s[C].reorder(C.axis[2], C.axis[1])
    s.buffer_at(C, s[C], C.axis[0])
    s[C].pipeline(C.axis[2])
    print(hcl.lower(s))

    f = hcl.build(s, "vhls")
    print(f)


if __name__ == "__main__":
    test_gemm_cpu()
    # test_gemm_fpga()
    # test_gemm_ihls()
    # test_gemm_outline()
    # test_gemm_buffer()
