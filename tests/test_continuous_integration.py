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

def test_vitis_sim():
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

def test_autosa_backend():
    def test_hls(size, target_mode):
        m = size
        n = size
        k = size

        dtype=hcl.Float()
        hcl.init(dtype)

        A = hcl.placeholder((m,k), dtype=dtype, name="A")
        B = hcl.placeholder((k,n), dtype=dtype, name="B")

        def kernel(A, B):
            Y = hcl.compute((m, n), lambda *args: 0, dtype=dtype, name="Y0")
            with hcl.Stage("Y"):
                with hcl.for_(0, m, name="i") as i:
                    with hcl.for_(0, n, name="j") as j:
                        Y[i][j] = 0
                        with hcl.for_(0, k, name="k") as r:
                            Y[i][j] += A[i][r] * B[r][j]
            return Y

        p = hcl.Platform.xilinx_zc706
        p.config(compiler="vitis", mode=target_mode)

        s = hcl.create_schedule([A, B], kernel)
        MM = kernel.Y

        s.to([A, B, kernel.Y0], p.xcel)
        s.to(kernel.Y.Y0, p.host)

        # intra-kernel data placement to create systolic araray
        s[kernel.Y].systolic()
        # using .to() as alternative to .systolic() for SA generation
        # PEs = s[kernel.Y].unroll(axis=[0,1], explicit=True)
        # for r in range(64):
        #     s.to(PEs[r,0].A, PEs[r,1]).to(PEs[r,2]).to(PEs[r,3])...
        # for c in range(64):
        #     s.to(PEs[0,c].B, PEs[1,c]).to(PEs[2,c]).to(PEs[3,c])...        

        s.transpose(kernel.Y.B)
        s.pack([MM.B, MM.A, MM.Y0], factor=512)

        np_A = np.random.randint(10, size=(m,k))
        np_B = np.random.randint(10, size=(k,n))
        np_C = np.zeros((m,n))
        args = (np_A, np_B, np_C)

        code = hcl.build(s, target=p)
        assert code.count("PE_wrapper") == 4098, code

    if os.getenv("LOCAL_CI_TEST"):
        if os.getenv("AUTOSA"):
            assert os.path.exists(os.getenv("AUTOSA")) 
            test_hls(1024, "debug")
    else:
        assert os.getenv("LOCAL_CI_TEST") == None

if __name__ == "__main__":
    test_vivado_hls()
    test_vitis_sim()
    test_autosa_backend()
