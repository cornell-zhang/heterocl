import heterocl as hcl
import numpy as np
import os

def test_vivado_hls():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    qtype = hcl.UInt(1)
    A = hcl.placeholder((10,10), "A", dtype=qtype)
    def kernel(A):
        return hcl.compute((10,10), lambda x, y: A[x][y] | 1, "B", dtype=qtype)
    s = hcl.create_schedule(A, kernel)
    target = hcl.platform.aws_f1
    target.config(compile="vivado_hls", mode="csim|csyn")
    s.to(A, target.xcel)
    s.to(kernel.B,target.host)

    f = hcl.build(s, target=target)
    np_A = np.random.random((10,10))
    np_B = np.zeros((10,10))

    hcl_A = hcl.asarray(np_A, dtype=qtype)
    hcl_B = hcl.asarray(np_B, dtype=qtype)
    f(hcl_A, hcl_B)

def test_vitis():
    if os.system("which v++ >> /dev/null") != 0:
        return 

    qtype = hcl.UInt(1)
    A = hcl.placeholder((10,10), "A", dtype=qtype)
    def kernel(A):
        return hcl.compute((10,10), lambda x, y: A[x][y] | 1, "B", dtype=qtype)
    s = hcl.create_schedule(A, kernel)
    target = hcl.platform.aws_f1
    target.config(compile="vitis", mode="hw_exe")
    s.to(A, target.xcel)
    s.to(kernel.B,target.host)

    f = hcl.build(s, target=target)
    np_A = np.random.random((10,10))
    np_B = np.zeros((10,10))

    hcl_A = hcl.asarray(np_A, dtype=qtype)
    hcl_B = hcl.asarray(np_B, dtype=qtype)
    passed = False
    try:
        f(hcl_A, hcl_B)
        passed = True
    except:
        pass
    assert passed == False

if __name__ == '__main__':
    # test_vitis()
    test_vivado_hls()
