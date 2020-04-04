import heterocl as hcl
import numpy as np
import numpy.testing as tst
import hlib
import os

dtype = hcl.Float(64)

def test_vector_add():

    def _test_llvm(in_shape):
        hcl.init(hcl.Float())
        A = hcl.placeholder(in_shape, name="A")
        B = hcl.placeholder(in_shape, name="B")

        def math_func(A, B):
            return hlib.op.extern.vector_add_rtl(A, B)

        s = hcl.create_schedule([A, B], math_func)
        f = hcl.build(s)

        _a = 10 * np.random.random(in_shape) - 5
        _b = 10 * np.random.random(in_shape) - 5
        out = hcl.asarray(np.zeros(in_shape).astype('float32'))

        real_out = np.add(_a, _b)
        f(hcl.asarray(_a), hcl.asarray(_b), out)
        tst.assert_almost_equal(out.asnumpy(), real_out, 4)

    _test_llvm((1, 3))
    _test_llvm((3, 3, 3))
    _test_llvm((5, 5, 3, 2))

    if (os.system("which v++ >> /dev/null") != 0):
      return 
  
    def _test_sim(in_shape):

        hcl.init(hcl.Int())
        A = hcl.placeholder(in_shape, name="A")
        B = hcl.placeholder(in_shape, name="B")

        def kernel(A, B):
            C = hlib.op.extern.scalar_add_rtl(A, B, name="C")
            return hcl.compute(in_shape, lambda *args: C[args] * 2, "D")

        target = hcl.platform.aws_f1
        target.config(compile="vitis", mode="sw_sim", backend="vhls")
        s = hcl.create_schedule([A, B], kernel)

        s.to([A, B], target.xcel)
        s.to(kernel.C, target.host)

        f = hcl.build(s, target)
        _a = 10 * np.random.random(in_shape) - 5
        _b = 10 * np.random.random(in_shape) - 5
        out = hcl.asarray(np.zeros(in_shape).astype('int32'))

        real_out = np.add(_a, _b) * 2
        f(hcl.asarray(_a), hcl.asarray(_b), out)
        tst.assert_almost_equal(out.asnumpy(), real_out, 4)

    _test_sim((1, 3))
    _test_sim((3, 3, 3))
    _test_sim((5, 5, 3, 2))

if __name__ == '__main__':
    test_vector_add()
