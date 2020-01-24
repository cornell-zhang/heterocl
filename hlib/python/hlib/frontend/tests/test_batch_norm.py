import heterocl as hcl
import hlib
import numpy as np
import numpy.testing as tst


def batch_norm_test(d_shape, axis=1):
    hcl.init(hcl.Float())
    data = hcl.placeholder(d_shape)
    if(axis < 0):
        axis += len(d_shape)
    shape = (d_shape[axis],)
    gamma = hcl.placeholder(shape)
    beta = hcl.placeholder(shape)
    moving_mean = hcl.placeholder(shape)
    moving_var = hcl.placeholder(shape)

    def func(data, gamma, beta, moving_mean, moving_var, axis=axis):
        return hlib.op.nn.batch_norm(
            data, gamma, beta, moving_mean, moving_var, axis=axis, epsilon=10**-5)[0]
    s = hcl.create_schedule([data, gamma, beta, moving_mean, moving_var], func)
    f = hcl.build(s)
    _data = np.random.randint(10, size=d_shape).astype(float)
    _gamma = np.ones(shape) * 2  # np.random.randint(10,size=shape)
    _beta = np.zeros(shape)  # np.random.randint(10,size=shape)
    _m_mean = np.random.randint(10, size=shape).astype(float)
    _m_var = np.random.randint(10, size=shape).astype(float)
    out = hcl.asarray(np.zeros(d_shape))
    m_mean = hcl.asarray(np.zeros(shape))
    m_var = hcl.asarray(np.zeros(shape))
    f(hcl.asarray(_data), hcl.asarray(_gamma), hcl.asarray(_beta),
      hcl.asarray(_m_mean), hcl.asarray(_m_var), out)

    def bn(a):
        np_sqrt = np.sqrt(_m_var + 10**-5)
        return (a - _m_mean)/np_sqrt * _gamma + _beta
    t_out = np.apply_along_axis(bn, axis, _data)
    tst.assert_almost_equal(t_out, out.asnumpy(), 3)

def test_batch_norm():
    batch_norm_test((3, 3), axis=0)
    batch_norm_test((3, 3), axis=1)
    batch_norm_test((2, 2, 2), axis=0)
    batch_norm_test((2, 2, 2), axis=1)
    batch_norm_test((2, 2, 2), axis=2)
