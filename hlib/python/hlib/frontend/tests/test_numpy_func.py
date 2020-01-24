import heterocl as hcl
import numpy as np
import hlib
import numpy.testing as tst
import tvm


def full_test(shape, fill_val=1, dtype=None):
    hcl.init(hcl.Float(32))
    def func(shape=shape, fill_val=fill_val, dtype=dtype):
        return hlib.op.math.full(shape, fill_val, dtype=dtype)
    s = hcl.create_schedule([], func)
    f = hcl.build(s)
    out = hcl.asarray(np.zeros(shape))
    real_out = np.full(shape, fill_val)
    f(out)
    return out.asnumpy(), real_out


def full_like_test(array_shape, fill_val=1, dtype=None):
    hcl.init(hcl.Float(32))
    array = hcl.placeholder(array_shape)

    def func(array, fill_val=fill_val):
        return hlib.op.math.full_like(array, fill_val)
    s = hcl.create_schedule(array, func)
    f = hcl.build(s)
    out = hcl.asarray(np.zeros(array_shape))
    _array = hcl.asarray(np.zeros(array_shape))
    real_out = np.full(array_shape, fill_val)
    f(_array, out)
    return out.asnumpy(), real_out


def zeros_test(shape, dtype=None):
    hcl.init(hcl.Float(32))
    def func(shape=shape, dtype=dtype):
        return hlib.op.math.zeros(shape, dtype=dtype)
    s = hcl.create_schedule([], func)
    f = hcl.build(s)
    shape = list(shape)
    for i in range(len(shape)):
        if hasattr(shape[i], 'value'):
            shape[i] = shape[i].value
    shape = tuple(shape)
    out = hcl.asarray(np.zeros(shape))
    real_out = np.zeros(shape)
    f(out)
    return out.asnumpy(), real_out


def zeros_like_test(array_shape, dtype=None):
    hcl.init(hcl.Float(32))
    array = hcl.placeholder(array_shape)

    def func(array):
        return hlib.op.math.zeros_like(array)
    s = hcl.create_schedule(array, func)
    f = hcl.build(s)
    out = hcl.asarray(np.zeros(array_shape))
    _array = hcl.asarray(np.zeros(array_shape))
    real_out = np.zeros(array_shape)
    f(_array, out)
    return out.asnumpy(), real_out


def ones_test(shape, dtype=None):
    hcl.init(hcl.Float(32))
    def func(shape=shape, dtype=dtype):
        return hlib.op.math.ones(shape, dtype=dtype)
    s = hcl.create_schedule([], func)
    f = hcl.build(s)
    out = hcl.asarray(np.zeros(shape))
    real_out = np.ones(shape)
    f(out)
    return out.asnumpy(), real_out


def ones_like_test(array_shape, dtype=None):
    hcl.init(hcl.Float(32))
    array = hcl.placeholder(array_shape)

    def func(array):
        return hlib.op.math.ones_like(array)
    s = hcl.create_schedule(array, func)
    f = hcl.build(s)
    out = hcl.asarray(np.zeros(array_shape))
    _array = hcl.asarray(np.zeros(array_shape))
    real_out = np.ones(array_shape)
    f(_array, out)
    return out.asnumpy(), real_out


def assert_gen(out, real_out):
    tst.assert_almost_equal(out, real_out, decimal=6)


def test_np_func():
    assert_gen(*full_test((3, 3), fill_val=5.01, dtype=hcl.Float()))
    assert_gen(*full_like_test((3, 3), fill_val=5.01, dtype=hcl.Float()))
    assert_gen(*zeros_test((3, 3), dtype=hcl.Float()))
    assert_gen(*zeros_test((1, 1), dtype=hcl.Float()))
    a = tvm.expr.IntImm('int', 1)
    assert_gen(*zeros_test((a, a), dtype=hcl.Float()))
    assert_gen(*zeros_like_test((3, 3), dtype=hcl.Float()))
    assert_gen(*ones_test((3, 3), dtype=hcl.Float()))
    assert_gen(*ones_like_test((3, 3), dtype=hcl.Float()))
