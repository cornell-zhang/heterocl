import heterocl as hcl
import hlib
import numpy as np
import numpy.testing as tst

def elemwise_test(in_size, func, dtype):
    hcl.init(dtype)
    in1 = hcl.placeholder(in_size)
    in2 = hcl.placeholder(in_size)

    def elem_func(in1, in2):
        return func(in1, in2)
    s = hcl.create_schedule([in1, in2], elem_func)
    f = hcl.build(s)
    _in1 = (4 * (np.random.random(in_size)) +
            1).astype(hcl.dtype_to_str(dtype))
    _in2 = (4 * (np.random.random(in_size)) +
            1).astype(hcl.dtype_to_str(dtype))
    out = hcl.asarray(np.zeros(in_size))
    f(hcl.asarray(_in1), hcl.asarray(_in2), out)
    return _in1, _in2, out.asnumpy()

def elemnot_test(in_size):
    hcl.init(hcl.Int())
    in1 = hcl.placeholder(in_size)

    def elem_func(in1):
        return hlib.op.op.logical_not(in1)
    s = hcl.create_schedule([in1], elem_func)
    f = hcl.build(s)
    _in = hcl.asarray(np.random.randint(1, high=10, size=in_size))
    out = hcl.asarray(np.zeros(in_size))
    f(_in, out)
    return _in.asnumpy(), out.asnumpy()

def elemint_test(in_shape, function):
    hcl.init(hcl.Int())
    input1 = hcl.placeholder(in_shape)
    input2 = hcl.placeholder(in_shape)

    sch_1 = hcl.create_schedule([input1, input2], function)
    test_1 = hcl.build(sch_1)

    _input1 = hcl.asarray(np.random.randint(1, high=5, size=in_shape))
    _input2 = hcl.asarray(np.random.randint(1, high=5, size=in_shape))
    _out = hcl.asarray(np.zeros(in_shape))
    test_1(_input1, _input2, _out)
    return _input1.asnumpy(), _input2.asnumpy(), _out.asnumpy()

def assert_add(in1, in2, out):
    tst.assert_almost_equal(in1 + in2, out)

def assert_sub(in1, in2, out):
    tst.assert_almost_equal(in1 - in2, out)

def assert_mul(in1, in2, out):
    tst.assert_almost_equal(in1 * in2, out)

def assert_div_int(in1, in2, out):
    tst.assert_almost_equal(in1 // in2, out)

def assert_div_float(in1, in2, out):
    tst.assert_almost_equal(in1 / in2, out)

def assert_mod(in1, in2, out):
    tst.assert_almost_equal(in1 % in2, out)

def assert_pow(in1, in2, out):
    tst.assert_almost_equal(pow(in1, in2), out, 3)

def assert_and(in1, in2, out):
    tst.assert_almost_equal((in1 & in2), out)

def assert_or(in1, in2, out):
    tst.assert_almost_equal((in1 | in2), out)

def assert_not(in1, out):
    tst.assert_almost_equal(~in1, out)

def test_add_int():
    assert_add(*elemwise_test((3,), hlib.op.op.elemwise_add, hcl.Int()))
    assert_add(*elemwise_test((3, 3), hlib.op.op.elemwise_add, hcl.Int()))
    assert_add(*elemwise_test((3, 3, 3), hlib.op.op.elemwise_add, hcl.Int()))
    assert_add(*elemwise_test((3, 3, 3, 3, 3), hlib.op.op.elemwise_add, hcl.Int()))

def test_add_float():
    assert_add(*elemwise_test((3,), hlib.op.op.elemwise_add, hcl.Float()))
    assert_add(*elemwise_test((3, 3), hlib.op.op.elemwise_add, hcl.Float()))
    assert_add(*elemwise_test((3, 3, 3), hlib.op.op.elemwise_add, hcl.Float()))
    assert_add(*elemwise_test((3, 3, 3, 3, 3),hlib.op.op.elemwise_add, hcl.Float()))

def test_sub_int():
    assert_sub(*elemwise_test((3,), hlib.op.op.elemwise_sub, hcl.Int()))
    assert_sub(*elemwise_test((3, 3), hlib.op.op.elemwise_sub, hcl.Int()))
    assert_sub(*elemwise_test((3, 3, 3), hlib.op.op.elemwise_sub, hcl.Int()))
    assert_sub(*elemwise_test((3, 3, 3, 3, 3), hlib.op.op.elemwise_sub, hcl.Int()))

def test_sub_float():
    assert_sub(*elemwise_test((3,), hlib.op.op.elemwise_sub, hcl.Float()))
    assert_sub(*elemwise_test((3, 3), hlib.op.op.elemwise_sub, hcl.Float()))
    assert_sub(*elemwise_test((3, 3, 3), hlib.op.op.elemwise_sub, hcl.Float()))
    assert_sub(*elemwise_test((3, 3, 3, 3, 3),hlib.op.op.elemwise_sub, hcl.Float()))

def test_mul_int():
    assert_mul(*elemwise_test((3,), hlib.op.op.elemwise_mul, hcl.Int()))
    assert_mul(*elemwise_test((3, 3), hlib.op.op.elemwise_mul, hcl.Int()))
    assert_mul(*elemwise_test((3, 3, 3), hlib.op.op.elemwise_mul, hcl.Int()))
    assert_mul(*elemwise_test((3, 3, 3, 3, 3), hlib.op.op.elemwise_mul, hcl.Int()))

def test_mul_float():
    assert_mul(*elemwise_test((3,), hlib.op.op.elemwise_mul, hcl.Float()))
    assert_mul(*elemwise_test((3, 3), hlib.op.op.elemwise_mul, hcl.Float()))
    assert_mul(*elemwise_test((3, 3, 3), hlib.op.op.elemwise_mul, hcl.Float()))
    assert_mul(*elemwise_test((3, 3, 3, 3, 3), hlib.op.op.elemwise_mul, hcl.Float()))

def test_div_int():
    assert_div_int(*elemwise_test((3,), hlib.op.op.elemwise_div, hcl.Int()))
    assert_div_int(*elemwise_test((3, 3), hlib.op.op.elemwise_div, hcl.Int()))
    assert_div_int(*elemwise_test((3, 3, 3), hlib.op.op.elemwise_div, hcl.Int()))
    assert_div_int(*elemwise_test((3, 3, 3, 3, 3), hlib.op.op.elemwise_div, hcl.Int()))

def test_div_float():
    assert_div_float(*elemwise_test((3,), hlib.op.op.elemwise_div, hcl.Float()))
    assert_div_float(*elemwise_test((3, 3), hlib.op.op.elemwise_div, hcl.Float()))
    assert_div_float(*elemwise_test((3, 3, 3),hlib.op.op.elemwise_div,hcl.Float()))
    assert_div_float(*elemwise_test((3, 3, 3, 3, 3), hlib.op.op.elemwise_div, hcl.Float()))

def test_mod_int():
    assert_mod(*elemwise_test((3,), hlib.op.op.elemwise_mod, hcl.Int()))
    assert_mod(*elemwise_test((3, 3), hlib.op.op.elemwise_mod, hcl.Int()))
    assert_mod(*elemwise_test((3, 3, 3), hlib.op.op.elemwise_mod, hcl.Int()))
    assert_mod(*elemwise_test((3, 3, 3, 3, 3), hlib.op.op.elemwise_mod, hcl.Int()))

def test_mod_float():
    assert_mod(*elemwise_test((3,), hlib.op.op.elemwise_mod, hcl.Float()))
    assert_mod(*elemwise_test((3, 3), hlib.op.op.elemwise_mod, hcl.Float()))
    assert_mod(*elemwise_test((3, 3, 3), hlib.op.op.elemwise_mod, hcl.Float()))
    assert_mod(*elemwise_test((3, 3, 3, 3, 3), hlib.op.op.elemwise_mod, hcl.Float()))

def test_pow_int():
    assert_pow(*elemwise_test((3,), hlib.op.op.elemwise_pow, hcl.Int()))
    assert_pow(*elemwise_test((3, 3), hlib.op.op.elemwise_pow, hcl.Int()))
    assert_pow(*elemwise_test((3, 3, 3), hlib.op.op.elemwise_pow, hcl.Int()))
    assert_pow(*elemwise_test((3, 3, 3, 3, 3), hlib.op.op.elemwise_pow, hcl.Int()))

def test_pow_float():
    assert_pow(*elemwise_test((3,), hlib.op.op.elemwise_pow, hcl.Float()))
    assert_pow(*elemwise_test((3, 3), hlib.op.op.elemwise_pow, hcl.Float()))
    assert_pow(*elemwise_test((3, 3, 3), hlib.op.op.elemwise_pow, hcl.Float()))
    assert_pow(*elemwise_test((3, 3, 3, 3, 3), hlib.op.op.elemwise_pow, hcl.Float()))

def test_and():
    assert_and(*elemint_test((3,), hlib.op.op.logical_and))
    assert_and(*elemint_test((3, 3), hlib.op.op.logical_and))
    assert_and(*elemint_test((3, 3, 3), hlib.op.op.logical_and))
    assert_and(*elemint_test((3, 3, 3, 3, 3), hlib.op.op.logical_and))

def test_or():
    assert_or(*elemint_test((3,), hlib.op.op.logical_or))
    assert_or(*elemint_test((3,3), hlib.op.op.logical_or))
    assert_or(*elemint_test((3,3,3), hlib.op.op.logical_or))
    assert_or(*elemint_test((3,3,3,3,3), hlib.op.op.logical_or))

def test_not():
    assert_not(*elemnot_test((3,)))
    assert_not(*elemnot_test((3,3)))
    assert_not(*elemnot_test((3,3,3)))
    assert_not(*elemnot_test((3,3,3,3,3)))
