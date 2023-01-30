import heterocl as hcl
import hlib
import numpy as np
import numpy.testing as tst


def bias_add_test(d_shape, b_shape, axis=1):
    hcl.init()
    data = hcl.placeholder(d_shape)
    bias = hcl.placeholder(b_shape)

    def func(data, bias, axis=axis):
        return hlib.op.nn.bias_add(data, bias, axis=axis)
    s = hcl.create_schedule([data, bias], func)
    f = hcl.build(s)
    _in = np.random.randint(10, size=d_shape)
    b = np.random.randint(10, size=b_shape)
    out = hcl.asarray(np.zeros(d_shape))
    f(hcl.asarray(_in), hcl.asarray(b), out)

    def add(a):
        return np.add(a, b)
    if axis < 0:
        axis += len(d_shape)
    t_out = np.apply_along_axis(add, axis, _in)
    tst.assert_almost_equal(t_out, out.asnumpy())

def test_bias_add():
    bias_add_test((3, 3, 3), (3,), -1)
    bias_add_test((3, 3, 3), (3,), 0)
    bias_add_test((3, 3, 3), (3,), 2)
    bias_add_test((3, 3, 3, 3), (3,), 0)
    bias_add_test((3, 3, 3, 3), (3,), 1)
    bias_add_test((3, 3, 3, 3), (3,), 2)
    bias_add_test((3, 3, 3, 3), (3,), -1)
