import heterocl as hcl
import numpy as np
import numpy.testing as tst
import hlib
import os
from itertools import permutations

dtype = hcl.Float(64)

def test_vector_add():

    def _test_llvm(length):
        hcl.init(hcl.Int())
        A = hcl.placeholder((length,), name="A")
        B = hcl.placeholder((length,), name="B")

        def math_func(A, B):
            return hlib.ip.vadd_rtl(A, B, length)

        s = hcl.create_schedule([A, B], math_func)
        f = hcl.build(s)

        np_A = np.random.randint(low=0, high=100, size=length)
        np_B = np.random.randint(low=0, high=100, size=length)
        np_out = np_A + np_B

        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B)
        
        hcl_out = hcl.asarray(np.zeros((length)))
        f(hcl_A, hcl_B, hcl_out)
        np.testing.assert_array_equal(np_out, hcl_out.asnumpy())

    _test_llvm(32)
    _test_llvm(512)
    _test_llvm(1024)

    if os.system("which v++ >> /dev/null") != 0:
        return 

    def _test_sim(length):
        hcl.init(hcl.Int())
        A = hcl.placeholder((length,), name="A")
        B = hcl.placeholder((length,), name="B")

        def math_func(A, B):
            res = hlib.ip.vadd_rtl(A, B, length)
            return hcl.compute(A.shape, lambda *args: res[args] * 2, "out")

        target = hcl.platform.aws_f1
        s = hcl.create_schedule([A, B], math_func)
        s.to([A, B], target.xcel)
        s.to(math_func.out, target.host)

        # test ir correctness 
        ir = str(hcl.lower(s))
        pattern = "test({}.channel, {}.channel, out.channel)"
        combination = [ pattern.format(*_) 
                for _ in list(permutations(["A", "B"])) ]
        assert any([_ in ir for _ in combination])

        # target.config(compile="vitis", mode="hw_sim")
        # f = hcl.build(s, target)

        # np_A = np.random.randint(low=0, high=100, size=length)
        # np_B = np.random.randint(low=0, high=100, size=length)
        # np_out = (np_A + np_B) * 2

        # hcl_A = hcl.asarray(np_A)
        # hcl_B = hcl.asarray(np_B)
        # 
        # hcl_out = hcl.asarray(np.zeros((length)))
        # f(hcl_A, hcl_B, hcl_out)
        # np.testing.assert_array_equal(np_out, hcl_out.asnumpy())

    _test_sim(32)
    _test_sim(512)
    _test_sim(1024)

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

    if os.system("which v++ >> /dev/null") != 0:
        return 

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
        target.config(compile="vitis", backend="vhls", mode="hw_sim")
        s.to([X_real, X_imag], target.xcel)
        s.to(math_func.abs, target.host)

        # test ir 
        ir = str(hcl.lower(s))
        pattern = "test({}.channel, {}.channel, abs.channel)"
        combination = [ pattern.format(*_) 
                for _ in list(permutations(["X_real", "X_imag"])) ]
        assert any([_ in ir for _ in combination])

        f = hcl.build(s, target)

        x_real_np = np.random.random((length))
        x_imag_np = np.random.random((length))
        x_np = x_real_np + 1j * x_imag_np
        
        out_np = np.fft.fft(x_np)
        out_real_np = out_np.real
        out_imag_np = out_np.imag
        
        x_real_hcl = hcl.asarray(x_real_np)
        x_imag_hcl = hcl.asarray(x_imag_np)
        
        out_abs_hcl = hcl.asarray(np.zeros((length)))

        f(x_real_hcl, x_imag_hcl, out_abs_hcl)
        out_abs_np = np.sqrt(out_imag_np **2 + out_real_np**2)
        np.testing.assert_allclose(out_abs_np, out_abs_hcl.asnumpy(), rtol=1e-02, atol=1e-3)

    _test_sim(32)
    _test_sim(512)
    _test_sim(1024)

def test_byte_swap_rtl():

    def test_llvm_(length):
        hcl.init(hcl.UInt(32))
        input_vec = hcl.placeholder((length,),  name="input")

        # assume gsize = lsize = 1
        def math_func(input_vec):
            new_vec = hlib.ip.byte_swap_rtl(input_vec)
            return hcl.compute(input_vec.shape, lambda *args: new_vec[args] + 1, name="ret")

        s = hcl.create_schedule([input_vec], math_func)

        x_np = np.random.randint(low=2**16, high=2**20, size=length)
        y_np = np.zeros((length))
        for i in range(length):
            y_np[i] = np.bitwise_and((1 << 32) - 1, np.bitwise_or(x_np[i] << 16, x_np[i] >> 16)) 
            y_np[i] = y_np[i] + 1

        f = hcl.build(s)
        x_hcl = hcl.asarray(x_np)
        
        y_hcl = hcl.asarray(np.zeros((length)))
        f(x_hcl, y_hcl)
        np.testing.assert_array_equal(y_np, y_hcl.asnumpy())

    test_llvm_(32)
    test_llvm_(512)
    test_llvm_(1024)

    if os.system("which aoc >> /dev/null") != 0:
        return 

    def test_sim_(length):
        hcl.init(hcl.UInt(32))
        input_vec = hcl.placeholder((length,),  name="input")

        # assume gsize = lsize = 1
        def math_func(input_vec):
            new_vec = hlib.ip.byte_swap_rtl(input_vec)
            return hcl.compute(input_vec.shape, lambda *args: new_vec[args] + 1, name="ret")

        s = hcl.create_schedule([input_vec], math_func)
        target = hcl.platform.vlab
        target.config(compile="aocl", mode="debug")

        s.to(input_vec, target.xcel)
        s.to(math_func.ret, target.host)

        # test debug mode (source code checking)
        code = hcl.build(s, target)
        assert "my_byteswap(input[k])" in code

        # test software emulation
        target.config(compile="aocl", mode="sw_sim")
        f = hcl.build(s, target)
        x_np = np.random.randint(low=2**16, high=2**20, size=length)
        y_np = np.zeros((length))

        x_hcl = hcl.asarray(x_np)
        y_hcl = hcl.asarray(np.zeros((length)))
        f(x_hcl, y_hcl)

        for i in range(length):
            y_np[i] = np.bitwise_and((1 << 32) - 1, np.bitwise_or(x_np[i] << 16, x_np[i] >> 16)) 
            y_np[i] = y_np[i] + 1
        np.testing.assert_array_equal(y_np, y_hcl.asnumpy())

        # test modelsim simulation
        target.config(compile="aocl", mode="hw_sim")
        f = hcl.build(s, target)
        x_np = np.random.randint(low=2**16, high=2**20, size=length)
        y_np = np.zeros((length))

        x_hcl = hcl.asarray(x_np)
        y_hcl = hcl.asarray(np.zeros((length)))
        f(x_hcl, y_hcl)
        f.report(target)

        for i in range(length):
            y_np[i] = np.bitwise_and((1 << 32) - 1, np.bitwise_or(x_np[i] << 16, x_np[i] >> 16)) 
            y_np[i] = y_np[i] + 1
        np.testing.assert_array_equal(y_np, y_hcl.asnumpy())

    test_sim_(32)
    test_sim_(512)
    test_sim_(1024)

if __name__ == '__main__':
    test_vector_add()
    test_fft_hls()
    test_byte_swap_rtl()
