import heterocl as hcl
import os
import numpy as np

def test_vivado_hls():
    def test_hls(target_mode):
        hcl.init(hcl.Int(16))
        A = hcl.placeholder((10,), "A")
        def kernel(A):
            B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
            return B
        
        target = hcl.Platform.aws_f1
        s = hcl.create_schedule([A], kernel)
        s.to(A, target.xcel)
        s.to(kernel.B, target.host)
        target.config(compiler="vivado_hls", mode=target_mode)
        f = hcl.build(s, target)

        np_A = np.random.randint(10, size=(10,))
        np_B = np.zeros((10,))

        hcl_A = hcl.asarray(np_A, dtype=hcl.Int(16))
        hcl_B = hcl.asarray(np_B, dtype=hcl.Int(16))
        f(hcl_A, hcl_B)
        ret_B = hcl_B.asnumpy()

        report = f.report()
        np.testing.assert_array_equal(ret_B, (np_A+1)*1)
    
    if os.getenv("LOCAL_CI_TEST"):
        test_hls("csim|csyn")
    else:
        assert os.getenv("LOCAL_CI_TEST") == None

def test_vitis():
    def test_hls(target_mode):
        hcl.init(hcl.Int(16))
        A = hcl.placeholder((10,), "A")
        def kernel(A):
            B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
            return B
        
        target = hcl.Platform.aws_f1
        s = hcl.create_schedule([A], kernel)
        s.to(A, target.xcel)
        s.to(kernel.B, target.host)
        target.config(compiler="vitis", mode=target_mode)
        f = hcl.build(s, target)

        np_A = np.random.randint(10, size=(10,))
        np_B = np.zeros((10,))

        hcl_A = hcl.asarray(np_A, dtype=hcl.Int(16))
        hcl_B = hcl.asarray(np_B, dtype=hcl.Int(16))
        f(hcl_A, hcl_B)
        ret_B = hcl_B.asnumpy()
        np.testing.assert_array_equal(ret_B, (np_A+1)*1)

    if os.getenv("LOCAL_CI_TEST"):
        test_hls("sw_sim")
    else:
        assert os.getenv("LOCAL_CI_TEST") == None

if __name__ == "__main__":
    test_vivado_hls()
    test_vitis()
