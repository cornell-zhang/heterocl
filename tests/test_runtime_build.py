import heterocl as hcl
from itertools import permutations
import os
import numpy as np

def test_debug_mode():

    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
        C = hcl.compute(A.shape, lambda *args : B[args] + 1, "C")
        D = hcl.compute(A.shape, lambda *args : C[args] * 2, "D")
        return D

    def test_sdaccel_debug():
        target = hcl.Platform.aws_f1
        s = hcl.create_schedule([A], kernel)
        s.to(kernel.B, target.xcel)
        s.to(kernel.C, target.host)
        target.config(compiler="sdaccel", mode="debug", backend="vhls")
        code = hcl.build(s, target)
        print(code)
        assert "cl::Kernel kernel(program, \"test\", &err)" in code

    def test_vhls_debug():
        target = hcl.Platform.xilinx_zc706
        s = hcl.create_schedule([A], kernel)
        s.to(kernel.B, target.xcel)
        s.to(kernel.C, target.host)
        target.config(compiler="vivado_hls", mode="debug")
        code = hcl.build(s, target)
        print(code)
        assert "test(int B[10][32], int C[10][32])" in code

    test_sdaccel_debug()
    test_vhls_debug()

def test_vivado_hls():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    def test_hls(target_mode):
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        def kernel(A):
            B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
            C = hcl.compute(A.shape, lambda *args : B[args] + 1, "C")
            D = hcl.compute(A.shape, lambda *args : C[args] * 2, "D")
            return D
        
        target = hcl.Platform.aws_f1
        s = hcl.create_schedule([A], kernel)
        s.to(kernel.B, target.xcel)
        s.to(kernel.C, target.host)
        target.config(compiler="vivado_hls", mode=target_mode)
        f = hcl.build(s, target)

        np_A = np.random.randint(10, size=(10,32))
        np_B = np.zeros((10,32))

        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))
        f(hcl_A, hcl_B)
        ret_B = hcl_B.asnumpy()

        if "csyn" in target_mode:
            report = f.report()
            assert "ReportVersion" in report
        elif "csim" in target_mode:
            np.testing.assert_array_equal(ret_B, (np_A+2)*2)

    test_hls("csim")
    test_hls("csyn")
    test_hls("csim|csyn")

def test_mixed_stream():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")

    def kernel(A, B):
        C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], "C")
        D = hcl.compute(C.shape, lambda i, j: C[i][j] * 2, "D")
        E = hcl.compute(C.shape, lambda i, j: D[i][j] * 3, "E")
        return E

    target = hcl.Platform.aws_f1
    s = hcl.create_schedule([A, B], kernel)

    s.to([A, B], target.xcel)
    s.to(kernel.D, target.host)
    s.to(kernel.C, s[kernel.D])

    target.config(compiler="vivado_hls", mode="csim")
    f = hcl.build(s, target)

    np_A = np.random.randint(10, size=(10,32))
    np_B = np.random.randint(10, size=(10,32))
    np_C = np.zeros((10,32))

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_C = hcl.asarray(np_C, dtype=hcl.Int(32))
    f(hcl_A, hcl_B, hcl_C)
    ret_C = hcl_C.asnumpy()
    np.testing.assert_array_equal(ret_C, (np_A + np_B) * 6)

def test_vitis():
    if os.system("which v++ >> /dev/null") != 0:
        return 

    def test_arith_ops():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        def kernel(A):
            B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
            C = hcl.compute(A.shape, lambda *args : B[args] + 1, "C")
            D = hcl.compute(A.shape, lambda *args : C[args] * 2, "D")
            return D
        
        target = hcl.Platform.aws_f1
        s = hcl.create_schedule([A], kernel)
        s.to(kernel.B, target.xcel)
        s.to(kernel.C, target.host)
        target.config(compiler="vitis", mode="hw_sim")
        f = hcl.build(s, target)

        np_A = np.random.randint(10, size=(10,32))
        np_B = np.zeros((10,32))

        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))
        f(hcl_A, hcl_B)
        ret_B = hcl_B.asnumpy()

        assert np.array_equal(ret_B, (np_A+2)*2)

    def test_xrt_stream():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        B = hcl.placeholder((10, 32), "B")

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda i, j: A[i,j] + B[i,j], "C")
            D = hcl.compute(C.shape, lambda i, j: C[i,j] + 1, "D")
            return D

        target = hcl.Platform.aws_f1
        target.config(compiler="vitis", mode="sw_sim")
        s = hcl.create_schedule([A, B], kernel)
        s.to(A, target.xcel, mode=hcl.IO.FIFO)
        s.to(B, target.xcel, mode=hcl.IO.DMA)
        s.to(kernel.D, target.host, mode=hcl.IO.FIFO)

        f = hcl.build(s, target)
        np_A = np.random.randint(10, size=(10,32))
        np_B = np.random.randint(10, size=(10,32))
        np_D = np.zeros((10,32))

        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))
        hcl_D = hcl.asarray(np_D)
        f(hcl_A, hcl_B, hcl_D)

        assert np.array_equal(hcl_D.asnumpy(), np_A + np_B + 1)

    test_arith_ops()
    test_xrt_stream()

def test_xilinx_sdsoc():
    if os.system("which sds++ >> /dev/null") != 0:
        return 

    def test_add_mul():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        def kernel(A):
            B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
            C = hcl.compute(A.shape, lambda *args : B[args] + 1, "C")
            D = hcl.compute(A.shape, lambda *args : C[args] * 2, "D")
            return D
        
        target = hcl.Platform.xilinx_zc706
        s = hcl.create_schedule([A], kernel)
        s.to(kernel.B, target.xcel)
        s.to(kernel.C, target.host)
        target.config(compiler="sdsoc", mode="sw_sim")
        f = hcl.build(s, target)

        np_A = np.random.randint(10, size=(10,32))
        np_B = np.zeros((10,32))

        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))
        f(hcl_A, hcl_B)

        assert np.array_equal(hcl_B.asnumpy(), np_A * 2 + 2)

    test_add_mul()

def test_intel_aocl():
    if os.system("which aocl >> /dev/null") != 0:
        return 

    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
        C = hcl.compute(A.shape, lambda *args : B[args] + 1, "C")
        D = hcl.compute(A.shape, lambda *args : C[args] * 2, "D")
        return D
    
    target = hcl.Platform.intel_vlab
    s = hcl.create_schedule([A], kernel)
    s.to(A, target.xcel)
    s.to(kernel.C, target.host)

    target.config(compiler="aocl", mode="sw_sim")
    f = hcl.build(s, target)

    np_A = np.random.randint(10, size=(10,32))
    np_B = np.zeros((10,32))

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))
    f(hcl_A, hcl_B)
    ret_B = hcl_B.asnumpy()

    np.testing.assert_array_equal(ret_B, (np_A + 2) * 2)

def test_project():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    dtype = hcl.Float()
    M = 64
    K = 64
    N = 64
    A = hcl.placeholder((M, K), "A", dtype=dtype)
    B = hcl.placeholder((K, N), "B", dtype=dtype)
    k = hcl.reduce_axis(0, K)
    def kernel(A, B):
        C = hcl.compute((M, N), lambda x, y: hcl.sum(A[x, k] * B[k, y], axis=k, dtype=dtype), "C", dtype=dtype)
        return C
    
    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="csyn", project="gemm")

    def make_schedule(opt=False):
        s = hcl.create_schedule([A, B], kernel, name=("s2" if opt else "s1"))
        s.to([A, B],target.xcel)
        s.to(kernel.C,target.host)

        def optimization():
            s[kernel.C].pipeline(kernel.C.axis[1])
            s.partition(A,hcl.Partition.Block,dim=2,factor=16)
            s.partition(B,hcl.Partition.Block,dim=1,factor=16)

        if opt:
            optimization()
        f = hcl.build(s, target)

        np_A = np.random.randint(0, 10, (M, K))
        np_B = np.random.randint(0, 10, (K, N))
        np_C = np.zeros((M, N))
        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B)
        hcl_C = hcl.asarray(np_C)
        f(hcl_A, hcl_B, hcl_C)
        return f

    f1 = make_schedule(opt=False)
    assert os.path.isdir("gemm-s1/out.prj")
    f2 = make_schedule(opt=True)
    assert os.path.isdir("gemm-s2/out.prj")

if __name__ == '__main__':
    test_debug_mode()
    test_vivado_hls()
    test_mixed_stream()
    test_vitis()
    test_xilinx_sdsoc()
    test_intel_aocl()
    test_project()
