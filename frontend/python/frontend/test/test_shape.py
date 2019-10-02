import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np
import hlib


def expand_dim_test(in_shape, axis, new_axis):
    input1 = hcl.placeholder(in_shape)

    def func(input1, axis=axis, new_axis=new_axis):
        return hlib.nn.expand_dims(input1, axis, new_axis)
    s = hcl.create_schedule([input1], func)
    f = hcl.build(s)
    _in = np.random.randint(50, size=in_shape)
    real_out = _in
    for i in range(new_axis):
        real_out = np.expand_dims(real_out, axis)

    def _new_shape(in_shape, axis, new_axis):
        new_shape = []
        for i in range(axis):
            new_shape.append(in_shape[i])
        for i in range(new_axis):
            new_shape.append(1)
        for i in range(len(in_shape) - axis):
            new_shape.append(in_shape[i + axis])
        return new_shape
    _out = hcl.asarray(np.zeros(_new_shape(in_shape, axis, new_axis)))
    _in = hcl.asarray(_in)
    f(_in, _out)
    return _in.asnumpy(), _out.asnumpy(), real_out


def squeeze_test(in_shape, axis=None):
    input1 = hcl.placeholder(in_shape)

    def func(input1, axis=axis):
        return hlib.nn.squeeze(input1, axis)
    s = hcl.create_schedule([input1], func)
    f = hcl.build(s)
    _in = np.random.randint(50, size=in_shape)
    real_out = _in
    real_out = np.squeeze(real_out, axis)

    def _new_shape(in_shape, axis):
        new_shape = []
        if(axis is None):
            for i in range(len(in_shape)):
                if in_shape[i] != 1:
                    new_shape.append(in_shape[i])
        else:
            for i in range(len(in_shape)):
                if i not in axis:
                    new_shape.append(in_shape[i])
        return new_shape
    _out = hcl.asarray(np.zeros(_new_shape(in_shape, axis)))
    _in = hcl.asarray(_in)
    f(_in, _out)
    return _in.asnumpy(), _out.asnumpy(), real_out


def split_test(in_shape, i_or_s, axis=0):
    input1 = hcl.placeholder(in_shape)

    def func(input1, i_or_s=i_or_s, axis=axis):
        return hlib.nn.split(input1, i_or_s, axis)
    s = hcl.create_schedule([input1], func)
    f = hcl.build(s)
    _in = np.random.randint(50, size=in_shape)
    real_out = np.split(_in, i_or_s, axis)
    new_shape = []
    for i in range(len(real_out)):
        new_shape.append(real_out[i].shape)
    _out = []
    if isinstance(i_or_s, list):
        num_outputs = len(i_or_s) + 1
    elif isinstance(i_or_s, int):
        num_outputs = i_or_s
    for i in range(num_outputs):
        _out.append(hcl.asarray(np.zeros(new_shape[i])))
    _in = hcl.asarray(_in)
    f(_in, *_out)
    for i in range(len(_out)):
        _out[i] = _out[i].asnumpy()
    return _in.asnumpy(), _out, real_out

def concat_test(data_tup_shape,axis=0):
    axis_len = 0
    input_tup = []
    for i in range(len(data_tup_shape)):
        axis_len += (data_tup_shape[i])[axis]
        input_tup.append(hcl.placeholder(data_tup_shape[i]))

    def func(*data_tup,axis=axis):
        return hlib.nn.concatenate(data_tup,axis=axis)
    s = hcl.create_schedule(input_tup,func)
    f = hcl.build(s)
    _in=[]
    for i in range(len(data_tup_shape)):
        _in.append(np.random.randint(50, size=data_tup_shape[i]))
    real_out = np.concatenate(tuple(_in),axis=axis)
    new_shape = list(data_tup_shape[0])
    new_shape[axis] = axis_len
    _out = hcl.asarray(np.zeros(tuple(new_shape)))
    for i in range(len(_in)):
        _in[i]=hcl.asarray(_in[i])
    f(*_in,_out)
    print(_out.asnumpy(),real_out)
    return _out.asnumpy(),real_out

def assert_expand_dim(_in, real_out, out):
    assert(np.array_equal(real_out, out))


def assert_squeeze(_in, real_out, out):
    assert(np.array_equal(real_out, out))

def assert_concatenate(real_out, out):
    assert(np.array_equal(real_out, out))

def assert_split(_in, out, real_out):
    for i in range(len(out)):
        assert(np.array_equal(real_out[i], out[i]))


assert_expand_dim(*expand_dim_test((3, 3), 1, 1))
assert_expand_dim(*expand_dim_test((3, 3, 3, 3, 3, 3), 2, 2))

assert_squeeze(*squeeze_test((1, 1, 3, 3, 3, 3)))
assert_squeeze(*squeeze_test((1, 1, 3, 3, 3, 3), axis=(1,)))
assert_squeeze(*squeeze_test((1, 1, 3, 3, 3, 3), axis=(0,)))
assert_squeeze(*squeeze_test((1, 1, 3, 3, 3, 3), axis=(1, 0)))

assert_split(*split_test((3, 4), 3, axis=0))
assert_split(*split_test((6, 3), [1, 3], axis=0))

assert_concatenate(*concat_test(((3,3),(4,3))))
assert_concatenate(*concat_test(((2,3),(4,3),(2,3),(2,3),(2,3))))
assert_concatenate(*concat_test(((2,3),(2,4)),axis=1))