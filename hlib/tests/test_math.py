import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np
import numpy.testing as tst
import hlib

dtype = hcl.Float(64)

_sum = hcl.reducer(0, lambda x, y: x + y, dtype)
_max = hcl.reducer(-100000, lambda x, y: tvm.make.Max(x, y), dtype)
_min = hcl.reducer(100000, lambda x, y: tvm.make.Min(x, y), dtype)
_prod = hcl.reducer(1, lambda x, y: x * y, dtype)

def test_exp():
    def _test(in_shape):
        hcl.init(hcl.Float())
        data = hcl.placeholder(in_shape)

        def math_func(data):
            return hlib.op.math.exp(data)
        s = hcl.create_schedule(data, math_func)
        f = hcl.build(s)
        _in = 10 * np.random.random(in_shape) - 5
        out = hcl.asarray(np.zeros(in_shape).astype('float32'))
        real_out = np.exp(_in)
        f(hcl.asarray(_in), out)
        tst.assert_almost_equal(out.asnumpy(), real_out, 4)

    _test((1, 3))
    _test((3, 3, 3))
    _test((5, 5, 3, 2))


def test_log():
    def _test(in_shape):
        hcl.init(hcl.Float())
        data = hcl.placeholder(in_shape)

        def math_func(data):
            return hlib.op.math.log(data)
        s = hcl.create_schedule(data, math_func)
        f = hcl.build(s)
        _in = 10 * np.random.random(in_shape) + 1
        out = hcl.asarray(np.zeros(in_shape).astype('float32'))
        real_out = np.log(_in)
        f(hcl.asarray(_in), out)
        tst.assert_almost_equal(out.asnumpy(), real_out, 5)

    _test((1, 3))
    _test((3, 3, 3))
    _test((5, 5, 3, 2))


def test_sigmoid():
    def _test(in_shape):
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
        tst.assert_almost_equal(out.asnumpy(), real_out, 5)

    _test((1, 3))
    _test((3, 3, 3))
    _test((5, 5, 3, 2))


def test_sqrt():
    def _test(in_shape):
        data = hcl.placeholder(in_shape)

        def math_func(data):
            return hlib.op.math.sqrt(data)
        s = hcl.create_schedule(data, math_func)
        f = hcl.build(s)
        _in = 100 * np.random.random(in_shape) + 1
        out = hcl.asarray(np.zeros(in_shape).astype('float32'))
        real_out = np.sqrt(_in)
        f(hcl.asarray(_in), out)
        tst.assert_almost_equal(out.asnumpy(), real_out, 5)

    _test((1, 3))
    _test((3, 3, 3))
    _test((5, 5, 3, 2))


def tanh_test():
    def _test(in_shape):
        hcl.init(hcl.Float())
        data = hcl.placeholder(in_shape)

        def math_func(data):
            return hlib.op.math.tanh(data)
        s = hcl.create_schedule(data, math_func)
        f = hcl.build(s)
        _in = 100 * np.random.random(in_shape) - 50
        out = hcl.asarray(np.zeros(in_shape).astype('float32'))
        real_out = np.tanh(_in)
        f(hcl.asarray(_in), out)
        tst.assert_almost_equal(out.asnumpy(), real_out, 5)

    _test((1, 3))
    _test((3, 3, 3))
    _test((5, 5, 3, 2))


def test_clip():
    def _test(in_shape, x_min, x_max):
        hcl.init(hcl.Float())
        data = hcl.placeholder(in_shape)

        def math_func(data, x_min=x_min, x_max=x_max):
            return hlib.op.math.clip(data, a_min=x_min, a_max=x_max)
        s = hcl.create_schedule(data, math_func)
        f = hcl.build(s)
        _in = 10 * np.random.random(in_shape) - 5
        out = hcl.asarray(np.zeros(in_shape).astype('float32'))
        real_out = np.clip(_in, x_min, x_max)
        f(hcl.asarray(_in), out)
        tst.assert_almost_equal(out.asnumpy(), real_out)

    _test((1, 3), 0, 4)
    _test((1, 3, 3), -4, 4)
    _test((1, 3), 0, 4)
    _test((3, 3), 0, 0.01)


def test_sum():
    def _test(in_shape, axis=None, keepdims=False):
        hcl.init()
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
            return hlib.op.math.sum(data, axis, keepdims)
        s = hcl.create_schedule(data, math_func)
        f = hcl.build(s)
        _in = np.random.randint(10, size=in_shape)
        if keepdims:
            out = hcl.asarray(np.zeros(new_shape))
        else:
            out = hcl.asarray(np.squeeze(np.zeros(new_shape)))
        f(hcl.asarray(_in), out)
        real_out = np.sum(_in, axis=axis, keepdims=keepdims)
        tst.assert_almost_equal(real_out, out.asnumpy())

    _test((3, 3), axis=(0,))
    _test((3, 3), axis=(0,), keepdims=True)
    _test((2, 2, 2), axis=(0,))
    _test((2, 2, 2), axis=(0,), keepdims=True)
    _test((2, 2, 2), axis=(1,))
    _test((2, 2, 2), axis=(1,), keepdims=True)
    _test((2, 2, 2), axis=(2,))
    _test((2, 2, 2), axis=(2,), keepdims=True)
    _test((2, 2, 2, 3), axis=(0,))
    _test((2, 2, 2, 3), axis=(0,), keepdims=True)
    _test((2, 2, 2, 3), axis=(1,))
    _test((2, 2, 2, 3), axis=(1,), keepdims=True)
    _test((2, 2, 2, 3), axis=(2,))
    _test((2, 2, 2, 3), axis=(2,), keepdims=True)
    _test((2, 2, 2, 3), axis=(3,))
    _test((2, 2, 2, 3), axis=(3,), keepdims=True)
    _test((2, 2, 2, 3), axis=(0, 1))
    _test((2, 2, 2, 3), axis=(0, 1), keepdims=True)
    _test((2, 2, 2, 3), axis=(0, 2))
    _test((2, 2, 2, 3), axis=(0, 2), keepdims=True)
    _test((5, 2, 4, 3), axis=(3,))
    _test((5, 2, 4, 3), axis=(3,), keepdims=True)
    _test((5, 2, 4, 3), axis=(0, 1))
    _test((5, 2, 4, 3), axis=(0, 1), keepdims=True)
    _test((5, 2, 4, 3), axis=(0, 2))
    _test((5, 2, 4, 3), axis=(0, 2), keepdims=True)


def test_max():
    def _test(in_shape, axis=None, keepdims=True):
        hcl.init()
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
        if keepdims:
            out = hcl.asarray(np.zeros(new_shape))
        else:
            out = hcl.asarray(np.squeeze(np.zeros(new_shape)))
        real_out = np.amax(_in, tuple(axis), keepdims=keepdims)
        f(hcl.asarray(_in), out)
        tst.assert_almost_equal(out.asnumpy(), real_out)

    _test((3, 3), axis=(0,))
    _test((3, 3), axis=(0,), keepdims=True)
    _test((2, 2, 2), axis=(0,))
    _test((2, 2, 2), axis=(0,), keepdims=True)
    _test((2, 2, 2), axis=(1,))
    _test((2, 2, 2), axis=(1,), keepdims=True)
    _test((2, 2, 2), axis=(2,))
    _test((2, 2, 2), axis=(2,), keepdims=True)
    _test((2, 2, 2, 3), axis=(0,))
    _test((2, 2, 2, 3), axis=(0,), keepdims=True)
    _test((2, 2, 2, 3), axis=(1,))
    _test((2, 2, 2, 3), axis=(1,), keepdims=True)
    _test((2, 2, 2, 3), axis=(2,))
    _test((2, 2, 2, 3), axis=(2,), keepdims=True)
    _test((2, 2, 2, 3), axis=(3,))
    _test((2, 2, 2, 3), axis=(3,), keepdims=True)
    _test((2, 2, 2, 3), axis=(0, 1))
    _test((2, 2, 2, 3), axis=(0, 1), keepdims=True)
    _test((2, 2, 2, 3), axis=(0, 2))
    _test((2, 2, 2, 3), axis=(0, 2), keepdims=True)
    _test((5, 2, 4, 3), axis=(3,))
    _test((5, 2, 4, 3), axis=(3,), keepdims=True)
    _test((5, 2, 4, 3), axis=(0, 1))
    _test((5, 2, 4, 3), axis=(0, 1), keepdims=True)
    _test((5, 2, 4, 3), axis=(0, 2))
    _test((5, 2, 4, 3), axis=(0, 2), keepdims=True)


def test_prod():
    def _test(in_shape, axis=None, keepdims=True):
        hcl.init(hcl.Float())
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
        if keepdims:
            out = hcl.asarray(np.zeros(new_shape))
        else:
            out = hcl.asarray(np.squeeze(np.zeros(new_shape)))
        real_out = np.prod(_in, tuple(axis), keepdims=keepdims)
        f(hcl.asarray(_in), out)
        tst.assert_almost_equal(out.asnumpy(), real_out)

    _test((3, 3), axis=(0,))
    _test((3, 3), axis=(0,), keepdims=True)
    _test((2, 2, 2), axis=(0,))
    _test((2, 2, 2), axis=(0,), keepdims=True)
    _test((2, 2, 2), axis=(1,))
    _test((2, 2, 2), axis=(1,), keepdims=True)
    _test((2, 2, 2), axis=(2,))
    _test((2, 2, 2), axis=(2,), keepdims=True)
    _test((2, 2, 2, 3), axis=(0,))
    _test((2, 2, 2, 3), axis=(0,), keepdims=True)
    _test((2, 2, 2, 3), axis=(1,))
    _test((2, 2, 2, 3), axis=(1,), keepdims=True)
    _test((2, 2, 2, 3), axis=(2,))
    _test((2, 2, 2, 3), axis=(2,), keepdims=True)
    _test((2, 2, 2, 3), axis=(3,))
    _test((2, 2, 2, 3), axis=(3,), keepdims=True)
    _test((2, 2, 2, 3), axis=(0, 1))
    _test((2, 2, 2, 3), axis=(0, 1), keepdims=True)
    _test((2, 2, 2, 3), axis=(0, 2))
    _test((2, 2, 2, 3), axis=(0, 2), keepdims=True)
    _test((5, 2, 4, 3), axis=(3,))
    _test((5, 2, 4, 3), axis=(3,), keepdims=True)
    _test((5, 2, 4, 3), axis=(0, 1))
    _test((5, 2, 4, 3), axis=(0, 1), keepdims=True)
    _test((5, 2, 4, 3), axis=(0, 2))
    _test((5, 2, 4, 3), axis=(0, 2), keepdims=True)


def test_min():
    def _test(in_shape, axis=None, keepdims=True):
        hcl.init()
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
        if keepdims:
            out = hcl.asarray(np.zeros(new_shape))
        else:
            out = hcl.asarray(np.squeeze(np.zeros(new_shape)))
        real_out = np.amin(_in, tuple(axis), keepdims=keepdims)
        f(hcl.asarray(_in), out)
        tst.assert_almost_equal(out.asnumpy(), real_out)

    _test((3, 3), axis=(0,))
    _test((3, 3), axis=(0,), keepdims=True)
    _test((2, 2, 2), axis=(0,))
    _test((2, 2, 2), axis=(0,), keepdims=True)
    _test((2, 2, 2), axis=(1,))
    _test((2, 2, 2), axis=(1,), keepdims=True)
    _test((2, 2, 2), axis=(2,))
    _test((2, 2, 2), axis=(2,), keepdims=True)
    _test((2, 2, 2, 3), axis=(0,))
    _test((2, 2, 2, 3), axis=(0,), keepdims=True)
    _test((2, 2, 2, 3), axis=(1,))
    _test((2, 2, 2, 3), axis=(1,), keepdims=True)
    _test((2, 2, 2, 3), axis=(2,))
    _test((2, 2, 2, 3), axis=(2,), keepdims=True)
    _test((2, 2, 2, 3), axis=(3,))
    _test((2, 2, 2, 3), axis=(3,), keepdims=True)
    _test((2, 2, 2, 3), axis=(0, 1))
    _test((2, 2, 2, 3), axis=(0, 1), keepdims=True)
    _test((2, 2, 2, 3), axis=(0, 2))
    _test((2, 2, 2, 3), axis=(0, 2), keepdims=True)
    _test((5, 2, 4, 3), axis=(3,))
    _test((5, 2, 4, 3), axis=(3,), keepdims=True)
    _test((5, 2, 4, 3), axis=(0, 1))
    _test((5, 2, 4, 3), axis=(0, 1), keepdims=True)
    _test((5, 2, 4, 3), axis=(0, 2))
    _test((5, 2, 4, 3), axis=(0, 2), keepdims=True)
