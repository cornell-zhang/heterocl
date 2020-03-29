import heterocl as hcl
import numpy as np
import numpy.testing as tst
import hlib

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
  
    def _test_sim(in_shape):

        hcl.init(hcl.Float())
        A = hcl.placeholder(in_shape, name="A")
        B = hcl.placeholder(in_shape, name="B")

        def math_func(A, B):
            return hlib.op.extern.vector_add_rtl(A, B)

        target = hcl.platform.aws_f1
        target.config(compile="vitis", mode="debug")
        s = hcl.create_schedule([A, B], math_func)
        f = hcl.build(s, target)
        # print(f)

        # _a = 10 * np.random.random(in_shape) - 5
        # _b = 10 * np.random.random(in_shape) - 5
        # out = hcl.asarray(np.zeros(in_shape).astype('int32'))

        # real_out = np.add(_a, _b)
        # f(hcl.asarray(_a), hcl.asarray(_b), out)
        # tst.assert_almost_equal(out.asnumpy(), real_out, 4)

    _test_sim((1, 3))
    _test_sim((3, 3, 3))
    _test_sim((5, 5, 3, 2))

if __name__ == '__main__':
    test_vector_add()
