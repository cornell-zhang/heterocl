import heterocl as hcl
from itertools import permutations
import os
import numpy as np

# FIXME: buffer mismatch for D
def test_placeholders():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")
    C = hcl.placeholder((10, 32), "C")
    D = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], "D")
    E = hcl.compute(C.shape, lambda i, j: C[i][j] * D[i][j], "E")
    F = hcl.compute(C.shape, lambda i, j: E[i][j] + 1, "F")

    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A, B, C, F])
    # s.to([A, B, C], target.xcel)
    # s.to(E, target.host)

    target.config(compile="sdaccel", backend="vhls")
    f = hcl.build(s, target)
    print(f)

def test_debug_mode():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
        C = hcl.compute(A.shape, lambda *args : B[args] + 1, "C")
        D = hcl.compute(A.shape, lambda *args : C[args] * 2, "D")
        return D
    
    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A], kernel)
    s.to(kernel.B, target.xcel)
    s.to(kernel.C, target.host)

    target.config(compile="sdaccel", mode="debug", backend="vhls")
    code = hcl.build(s, target)
    print(code)
    assert "cl::Kernel kernel(program, \"test\", &err)" in code


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
        
        target = hcl.platform.aws_f1
        s = hcl.create_schedule([A], kernel)
        s.to(kernel.B, target.xcel)
        s.to(kernel.C, target.host)
        target.config(compile="vivado_hls", mode=target_mode)
        f = hcl.build(s, target)

        np_A = np.random.randint(10, size=(10,32))
        np_B = np.zeros((10,32))

        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))
        f(hcl_A, hcl_B)
        ret_B = hcl_B.asnumpy()

        if "csyn" in target_mode:
            report = f.report("csyn")
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

    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A, B], kernel)
    s.to([A, B], target.xcel)
    s.to(kernel.D, target.host)
    s.to(kernel.C, s[kernel.D])

    target.config(compile="vivado_hls", mode="csim")
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

    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
        C = hcl.compute(A.shape, lambda *args : B[args] + 1, "C")
        D = hcl.compute(A.shape, lambda *args : C[args] * 2, "D")
        return D
    
    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A], kernel)
    s.to(kernel.B, target.xcel)
    s.to(kernel.C, target.host)
    target.config(compile="vitis", mode="sw_sim", backend="vhls")
    f = hcl.build(s, target)

    np_A = np.random.randint(10, size=(10,32))
    np_B = np.zeros((10,32))

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))
    f(hcl_A, hcl_B)
    ret_B = hcl_B.asnumpy()

    np.testing.assert_array_equal(ret_B, (np_A + 2) * 2)

def test_xilinx_sdsoc():
    if os.system("which sds++ >> /dev/null") != 0:
        return 

    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
        C = hcl.compute(A.shape, lambda *args : B[args] + 1, "C")
        D = hcl.compute(A.shape, lambda *args : C[args] * 2, "D")
        return D
    
    target = hcl.platform.zc706
    s = hcl.create_schedule([A], kernel)
    s.to(kernel.B, target.xcel)
    s.to(kernel.C, target.host)
    target.config(compile="sdsoc", mode="sw_sim")
    f = hcl.build(s, target)

    np_A = np.random.randint(10, size=(10,32))
    np_B = np.zeros((10,32))

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))
    f(hcl_A, hcl_B)
    ret_B = hcl_B.asnumpy()

    np.testing.assert_array_equal(ret_B, (np_A + 2) * 2)

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
    
    target = hcl.platform.vlab
    s = hcl.create_schedule([A], kernel)
    s.to(kernel.B, target.xcel)
    s.to(kernel.C, target.host)
    target.config(compile="aocl", mode="sw_sim")
    f = hcl.build(s, target)

    np_A = np.random.randint(10, size=(10,32))
    np_B = np.zeros((10,32))

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))
    f(hcl_A, hcl_B)
    ret_B = hcl_B.asnumpy()

    np.testing.assert_array_equal(ret_B, (np_A + 2) * 2)

if __name__ == '__main__':
    test_placeholders()
    test_debug_mode()
    test_vivado_hls()
    test_mixed_stream()
    test_vitis()
    test_xilinx_sdsoc()
    test_intel_aocl()
