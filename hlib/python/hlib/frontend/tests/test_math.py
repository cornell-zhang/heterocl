import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np
import numpy.testing as tst
import hlib

dtype = hcl.Float(64)
hcl.init(hcl.Float(64))

_sum = hcl.reducer(0, lambda x, y: x + y, dtype)
_max = hcl.reducer(-100000, lambda x, y: tvm.make.Max(x, y), dtype)
_min = hcl.reducer(100000, lambda x, y: tvm.make.Min(x, y), dtype)
_prod = hcl.reducer(1, lambda x, y: x * y, dtype)


def exp_test(in_shape):
    data = hcl.placeholder(in_shape)

    def math_func(data):
        return hlib.op.math.exp(data)
    s = hcl.create_schedule(data, math_func)
    f = hcl.build(s)
    _in = 10 * np.random.random(in_shape) - 5
    out = hcl.asarray(np.zeros(in_shape).astype('float32'))
    real_out = np.exp(_in)
    f(hcl.asarray(_in), out)
    tst.assert_almost_equal(out.asnumpy(), real_out)


def log_test(in_shape):
    data = hcl.placeholder(in_shape)

    def math_func(data):
        return hlib.op.math.log(data)
    s = hcl.create_schedule(data, math_func)
    f = hcl.build(s)
    _in = 100 * np.random.random(in_shape) + 1
    out = hcl.asarray(np.zeros(in_shape).astype('float32'))
    real_out = np.log(_in)
    f(hcl.asarray(_in), out)
    tst.assert_almost_equal(out.asnumpy(), real_out)


def sigmoid_test(in_shape):
    data = hcl.placeholder(in_shape)

    def math_func(data):
        return hlib.op.math.sigmoid(data)
    s = hcl.create_schedule(data, math_func)
    f = hcl.build(s)
    _in = 10 * np.random.random(in_shape) - 5
    out = hcl.asarray(np.zeros(in_shape).astype('float32'))

    def sigmoid(data):
        return 1 / (1 + np.exp(-data))
    real_out = sigmoid(_in)
    f(hcl.asarray(_in), out)
    tst.assert_almost_equal(out.asnumpy(), real_out)


def sqrt_test(in_shape):
    data = hcl.placeholder(in_shape)

    def math_func(data):
        return hlib.op.math.sqrt(data)
    s = hcl.create_schedule(data, math_func)
    f = hcl.build(s)
    _in = 100 * np.random.random(in_shape) + 1
    out = hcl.asarray(np.zeros(in_shape).astype('float32'))
    real_out = np.sqrt(_in)
    f(hcl.asarray(_in), out)
    tst.assert_almost_equal(out.asnumpy(), real_out)


def tanh_test(in_shape):
    data = hcl.placeholder(in_shape)

    def math_func(data):
        return hlib.op.math.tanh(data)
    s = hcl.create_schedule(data, math_func)
    f = hcl.build(s)
    _in = 100 * np.random.random(in_shape) - 50
    out = hcl.asarray(np.zeros(in_shape).astype('float32'))
    real_out = np.tanh(_in)
    f(hcl.asarray(_in), out)
    tst.assert_almost_equal(out.asnumpy(), real_out)


def clip_test(in_shape, x_min, x_max):
    data = hcl.placeholder(in_shape)

    def math_func(data, x_min=x_min, x_max=x_max):
        return hlib.op.math.clip(data, a_min=x_min, a_max=x_max)
    s = hcl.create_schedule(data, math_func)
    f = hcl.build(s)
    _in = 10 * np.random.random(in_shape) - 5
    out = hcl.asarray(np.zeros(in_shape).astype('float32'))
    real_out = np.clip(_in, x_min, x_max)
    f(hcl.asarray(_in), out)
    print(out.asnumpy(), real_out)
    tst.assert_almost_equal(out.asnumpy(), real_out)


def sum_test(in_shape, axis=None, keepdims=False):
    new_shape = []
    if axis is None:
        for i in range(len(in_shape)):
            new_shape.append(1)
    else:
        if isinstance(axis, int):
            if axis < 0:
                axis = len(in_shape) + axis
            axis = [axis]
        for i in range(len(in_shape)):
            if i in axis and keepdims:
                new_shape.append(1)
            else:
                new_shape.append(in_shape[i])
    axis_len = len(axis)
    _new_shape = []
    for i in range(len(in_shape)):
        if i not in axis:
            _new_shape.append(in_shape[i])
    while len(_new_shape) < len(in_shape):
        _new_shape.append(1)
    data = hcl.placeholder(in_shape)

    def math_func(data, axis=axis, keepdims=keepdims):
        return hlib.op.math.sum(data, axis, keepdims)
    s = hcl.create_schedule(data, math_func)
    f = hcl.build(s)
    _in = np.random.randint(10, size=in_shape)
    if keepdims:
        out = hcl.asarray(np.zeros(new_shape))
    else:
        out = hcl.asarray(np.squeeze(np.zeros(_new_shape)))
    f(hcl.asarray(_in), out)
    real_out = np.sum(_in, axis=axis, keepdims=keepdims)
    print(real_out.shape, out.shape)
    print(real_out)
    print(out.asnumpy())
    tst.assert_almost_equal(real_out, out.asnumpy())
    return _in, out.asnumpy()


def max_test(in_shape, axis=None, keepdims=True):
    new_shape = []
    if axis is None:
        for i in range(len(in_shape)):
            new_shape.append(1)
    else:
        if isinstance(axis, int):
            if axis < 0:
                axis = len(in_shape) + axis
            axis = [axis]
        for i in range(len(in_shape)):
            if i in axis:
                new_shape.append(1)
            else:
                new_shape.append(in_shape[i])
    data = hcl.placeholder(in_shape)

    def math_func(data, axis=axis, keepdims=keepdims):
        return hlib.op.math.max(data, axis, keepdims)
    s = hcl.create_schedule(data, math_func)
    f = hcl.build(s)
    _in = np.random.randint(10, size=in_shape)
    out = hcl.asarray(np.zeros(new_shape))
    f(hcl.asarray(_in), out)
    return _in, out.asnumpy()


def prod_test(in_shape, axis=None, keepdims=True):
    new_shape = []
    if axis is None:
        for i in range(len(in_shape)):
            new_shape.append(1)
    else:
        if isinstance(axis, int):
            if axis < 0:
                axis = len(in_shape) + axis
            axis = [axis]
        for i in range(len(in_shape)):
            if i in axis:
                new_shape.append(1)
            else:
                new_shape.append(in_shape[i])
    data = hcl.placeholder(in_shape)

    def math_func(data, axis=axis, keepdims=keepdims):
        return hlib.op.math.prod(data, axis, keepdims)
    s = hcl.create_schedule(data, math_func)
    f = hcl.build(s)
    _in = np.random.random(size=in_shape)
    out = hcl.asarray(np.zeros(new_shape))
    f(hcl.asarray(_in), out)
    return _in, out.asnumpy()


def min_test(in_shape, axis=None, keepdims=False):
    new_shape = []
    if axis is None:
        for i in range(len(in_shape)):
            new_shape.append(1)
    else:
        if isinstance(axis, int):
            if axis < 0:
                axis = len(in_shape) + axis
            axis = [axis]
        for i in range(len(in_shape)):
            if i in axis:
                new_shape.append(1)
            else:
                new_shape.append(in_shape[i])
    data = hcl.placeholder(in_shape)

    def math_func(data, axis=axis, keepdims=keepdims):
        return hlib.op.math.min(data, axis, keepdims)
    s = hcl.create_schedule(data, math_func)
    f = hcl.build(s)
    _in = np.random.randint(10, size=in_shape)
    out = hcl.asarray(np.squeeze(np.zeros(new_shape)))
    print(_in)
    f(hcl.asarray(_in), out)
    return _in, out.asnumpy()


clip_test((1, 3), 0, 4)
clip_test((1, 3, 3), -4, 4)
clip_test((1, 3), 0, 4)
clip_test((3, 3), 0, 0.01)

exp_test((1, 3))
exp_test((3, 3, 3))
exp_test((5, 5, 3, 2))

log_test((1, 3))
log_test((3, 3, 3))
log_test((5, 5, 3, 2))

sigmoid_test((1, 3))
sigmoid_test((3, 3, 3))
sigmoid_test((5, 5, 3, 2))

sqrt_test((1, 3))
sqrt_test((3, 3, 3))
sqrt_test((5, 5, 3, 2))

tanh_test((1, 3))
tanh_test((3, 3, 3))
tanh_test((5, 5, 3, 2))

sum_test((3,3),axis=(0,))
sum_test((2,2,2),axis=(0,))
sum_test((2,2,2),axis=(1,))
sum_test((2,2,2),axis=(2,))
sum_test((2,2,2,3),axis=(0,))
sum_test((2,2,2,3),axis=(1,))
sum_test((2,2,2,3),axis=(2,))
sum_test((2,2,2,3),axis=(3,))
sum_test((2,2,2,3),axis=(0,1))
sum_test((2,2,2,3),axis=(0,2))
sum_test((5,2,4,3),axis=(3,))
sum_test((5,4,2,3),axis=(0,1))
sum_test((5,2,4,3),axis=(0,2))
"""sum_test((3,3),axis=(0,),keepdims=True)
sum_test((2,2,2),axis=(0,),keepdims=True)
sum_test((2,2,2),axis=(1,),keepdims=True)
sum_test((2,2,2),axis=(2,),keepdims=True)
sum_test((2, 2, 2, 3), axis=(0,), keepdims=True)
sum_test((2, 2, 2, 3), axis=(1,), keepdims=True)
sum_test((2, 2, 2, 3), axis=(2,), keepdims=True)
sum_test((2, 2, 2, 3), axis=(3,), keepdims=True)
sum_test((2, 2, 2, 3), axis=(0, 1), keepdims=True)
sum_test((2, 2, 2, 3), axis=(0, 2), keepdims=True)
sum_test((5, 2, 4, 3), axis=(3,), keepdims=True)
sum_test((5, 4, 2, 3), axis=(0, 1), keepdims=True)
sum_test((5, 2, 4, 3), axis=(0, 2), keepdims=True)
sum_test((2, 3, 4), axis=(0, 2), keepdims=True)"""


print(max_test((3, 3), axis = (0,)))
print(max_test((2, 2, 2), axis=(0,)))

print(prod_test((3, 3), axis = (0,)))
print(prod_test((2, 2, 2), axis=(0,)))

print(min_test((3, 3), axis = (0,)))
print(min_test((2, 2, 2), axis=(0,)))
