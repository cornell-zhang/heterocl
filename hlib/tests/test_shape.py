import heterocl as hcl
import numpy as np
import hlib


def expand_dim_test(in_shape, axis, new_axis):
    hcl.init()
    input1 = hcl.placeholder(in_shape)

    def func(input1, axis=axis, new_axis=new_axis):
        return hlib.op.nn.expand_dims(input1, axis, new_axis)
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
    hcl.init()
    input1 = hcl.placeholder(in_shape)

    def func(input1, axis=axis):
        return hlib.op.nn.squeeze(input1, axis)
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
    hcl.init()
    input1 = hcl.placeholder(in_shape)

    def func(input1, i_or_s=i_or_s, axis=axis):
        return hlib.op.nn.split(input1, i_or_s, axis)
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


def concat_test(data_tup_shape, axis=0):
    hcl.init()
    axis_len = 0
    input_tup = []
    for i in range(len(data_tup_shape)):
        axis_len += (data_tup_shape[i])[axis]
        input_tup.append(hcl.placeholder(data_tup_shape[i]))

    def func(*data_tup, axis=axis):
        return hlib.op.nn.concatenate(*data_tup, axis=axis)
    s = hcl.create_schedule(input_tup, func)
    f = hcl.build(s)
    _in = []
    for i in range(len(data_tup_shape)):
        _in.append(np.random.randint(50, size=data_tup_shape[i]))
    real_out = np.concatenate(tuple(_in), axis=axis)
    new_shape = list(data_tup_shape[0])
    new_shape[axis] = axis_len
    _out = hcl.asarray(np.zeros(tuple(new_shape)))
    for i in range(len(_in)):
        _in[i] = hcl.asarray(_in[i])
    f(*_in, _out)
    return _out.asnumpy(), real_out


def red_mul(l):
    result = 1
    for item in l:
        result = result * item
    return result


def reshape_test(data_shape, newshape):
    hcl.init()
    input_shape = hcl.placeholder(data_shape)

    def func(data, new_shape=newshape):
        return hlib.op.nn.reshape(data, newshape=newshape)
    s = hcl.create_schedule(input_shape, func)
    f = hcl.build(s)
    _in = np.random.randint(50, size=data_shape)
    res_shape = []
    cur_shape = list(data_shape)
    inx = 0
    inx_n1 = -1
    val_n1 = 1
    for _ in range(len(newshape)):
        new_inx = newshape[inx]
        assert(new_inx > -5), "inx has to be greater than -5"
        if(new_inx > 0):
            res_shape.append(new_inx)
        elif(new_inx == 0):
            res_shape.append(cur_shape[inx])
        elif(new_inx == -1):
            if(not inx_n1 == -1):
                raise ValueError("no more than one -1 is allowed in newshape")
            inx_n1 = inx
        elif(new_inx == -2):
            res_shape.extend(cur_shape[inx:])
        elif(new_inx == -3):
            res_shape.append(cur_shape[inx] + cur_shape[inx + 1])
            inx = inx + 1
        elif(new_inx == -4):
            assert False, "not implemented yet"
        inx = inx + 1
    if(not inx_n1 == -1):
        res_shape.insert(inx_n1, red_mul(cur_shape) // red_mul(res_shape))
    out = hcl.asarray(np.zeros(tuple(res_shape)))
    f(hcl.asarray(_in), out)
    out = out.asnumpy()
    real_out = np.reshape(_in, res_shape)
    return out, real_out


def assert_expand_dim(_in, real_out, out):
    assert(np.array_equal(real_out, out))


def assert_squeeze(_in, real_out, out):
    assert(np.array_equal(real_out, out))


def assert_concatenate(real_out, out):
    assert(np.array_equal(real_out, out))


def assert_split(_in, out, real_out):
    for i in range(len(out)):
        assert(np.array_equal(real_out[i], out[i]))


def assert_reshape(real_out, out):
    assert(np.array_equal(real_out, out))


def test_expand_dim():
    assert_expand_dim(*expand_dim_test((3, 3), 1, 1))
    assert_expand_dim(*expand_dim_test((3, 3, 3, 3, 3, 3), 2, 2))


def test_squeeze():
    assert_squeeze(*squeeze_test((1, 1, 3, 3, 3, 3)))
    assert_squeeze(*squeeze_test((1, 1, 3, 3, 3, 3), axis=(1,)))
    assert_squeeze(*squeeze_test((1, 1, 3, 3, 3, 3), axis=(0,)))
    assert_squeeze(*squeeze_test((1, 1, 3, 3, 3, 3), axis=(1, 0)))

def test_split():
    assert_split(*split_test((3, 4), 3, axis=0))
    assert_split(*split_test((6, 3), [1, 3], axis=0))

def test_concat():
    assert_concatenate(*concat_test(((3, 3), (4, 3))))
    assert_concatenate(*concat_test(((2, 3), (4, 3), (2, 3), (2, 3), (2, 3))))
    assert_concatenate(*concat_test(((2, 3), (2, 4)), axis=1))
    assert_concatenate(*concat_test(((1, 2, 2), (1, 2, 2)), axis=2))

def test_reshape():
    assert_reshape(*reshape_test((9,), (3, 3)))
    assert_reshape(*reshape_test((2, 2, 2), (4, 2)))
    assert_reshape(*reshape_test((3, 2, 4), (6, 4, 1)))
    assert_reshape(*reshape_test((12,), (3, -1)))
    assert_reshape(*reshape_test((24,), (3, 2, 4)))
