import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np
import hlib
hcl.init(hcl.Float())


def batch_norm_test(d_shape, axis=1):
    data = hcl.placeholder(d_shape)
    if(axis < 0):
        axis = len(d_shape) - 1
    shape = (d_shape[axis],)
    gamma = hcl.placeholder(shape)
    beta = hcl.placeholder(shape)
    moving_mean = hcl.placeholder(shape)
    moving_var = hcl.placeholder(shape)

    def func(data, gamma, beta, moving_mean, moving_var, axis=axis):
        return hlib.op.nn.batch_norm(
            data, gamma, beta, moving_mean, moving_var, axis=axis, epsilon=10**-5)
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
      hcl.asarray(_m_mean), hcl.asarray(_m_var), out, m_mean, m_var)


batch_norm_test((3, 3), axis=0)
batch_norm_test((3, 3), axis=1)
batch_norm_test((2, 2, 2), axis=0)
batch_norm_test((2, 2, 2), axis=1)
batch_norm_test((2, 2, 2), axis=2)
