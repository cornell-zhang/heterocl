import heterocl as hcl
import numpy as np
import numpy.testing as tst
import hlib
import os
from itertools import permutations

dtype = hcl.Float(64)

def test_fft_hls():

    def _test_llvm(length):
        hcl.init(hcl.Float())
        X_real = hcl.placeholder((length,), name="X_real")
        X_imag = hcl.placeholder((length,),  name="X_imag")

        def math_func(A, B):
            return hlib.ip.single_fft_hls(A, B)

        s = hcl.create_schedule([X_real, X_imag], math_func)
        f = hcl.build(s)

        x_real_np = np.random.random((length))
        x_imag_np = np.random.random((length))
        x_np = x_real_np + 1j * x_imag_np
        
        out_np = np.fft.fft(x_np)
        out_real_np = out_np.real
        out_imag_np = out_np.imag
        
        x_real_hcl = hcl.asarray(x_real_np)
        x_imag_hcl = hcl.asarray(x_imag_np)
        
        out_real_hcl = hcl.asarray(np.zeros((length)))
        out_imag_hcl = hcl.asarray(np.zeros((length)))

        f(x_real_hcl, x_imag_hcl, out_real_hcl, out_imag_hcl)

        np.testing.assert_allclose(out_real_np, out_real_hcl.asnumpy(), rtol=1e-02, atol=1e-3)
        np.testing.assert_allclose(out_imag_np, out_imag_hcl.asnumpy(), rtol=1e-02, atol=1e-3)

    _test_llvm(32)
    _test_llvm(512)
    _test_llvm(1024)

    def _test_sim(length):
        hcl.init(hcl.Float())
        X_real = hcl.placeholder((length,), name="X_real")
        X_imag = hcl.placeholder((length,),  name="X_imag")

        def math_func(A, B):
            real, imag = hlib.ip.single_fft_hls(A, B)
            return hcl.compute((length,), lambda x: 
                    hcl.sqrt(real[x] * real[x] + imag[x] * imag[x]), name="abs")

        s = hcl.create_schedule([X_real, X_imag], math_func)
        target = hcl.platform.aws_f1
        target.config(compile="vitis", backend="vhls")
        s.to([X_real, X_imag], target.xcel)
        s.to(math_func.abs, target.host)
        ir = str(hcl.lower(s))
        pattern = "test({}.channel, {}.channel, abs.channel)"
        combination = [ pattern.format(*_) 
                for _ in list(permutations(["X_real", "X_imag"])) ]
        assert any([_ in ir for _ in combination])
        # f = hcl.build(s, target)

        # x_real_np = np.random.random((length))
        # x_imag_np = np.random.random((length))
        # x_np = x_real_np + 1j * x_imag_np
        # 
        # out_np = np.fft.fft(x_np)
        # out_real_np = out_np.real
        # out_imag_np = out_np.imag
        # 
        # x_real_hcl = hcl.asarray(x_real_np)
        # x_imag_hcl = hcl.asarray(x_imag_np)
        # 
        # out_real_hcl = hcl.asarray(np.zeros((length)))
        # out_imag_hcl = hcl.asarray(np.zeros((length)))

        # f(x_real_hcl, x_imag_hcl, out_real_hcl, out_imag_hcl)

        # np.testing.assert_allclose(out_real_np, out_real_hcl.asnumpy(), rtol=1e-02, atol=1e-3)
        # np.testing.assert_allclose(out_imag_np, out_imag_hcl.asnumpy(), rtol=1e-02, atol=1e-3)

    _test_sim(32)
    _test_sim(512)
    _test_sim(1024)

if __name__ == '__main__':
    test_fft_hls()
