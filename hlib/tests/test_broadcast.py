import heterocl as hcl
import hlib
import numpy as np
hcl.init(hcl.Float())

# function definitions


def broadcast_test(in_shape1, in_shape2, function):
    hcl.init()
    input1 = hcl.placeholder(in_shape1)
    input2 = hcl.placeholder(in_shape2)

    sch_1 = hcl.create_schedule([input1, input2], function)
    test_1 = hcl.build(sch_1)

    _input1 = hcl.asarray(np.random.randint(1, high=10, size=in_shape1))
    _input2 = hcl.asarray(np.random.randint(1, high=10, size=in_shape2))
    _out = hcl.asarray(np.zeros(in_shape1))
    test_1(_input1, _input2, _out)
    return _input1.asnumpy(), _input2.asnumpy(), _out.asnumpy()


def assert_b_add(in1, in2, out):
    assert(np.array_equal(in1 + in2, out))


def assert_b_sub(in1, in2, out):
    assert(np.array_equal(in1 - in2, out))


def assert_b_mul(in1, in2, out):
    assert(np.array_equal(in1 * in2, out))


def assert_b_div(in1, in2, out):
    assert(np.allclose(in1 // in2, out))


def assert_b_mod(in1, in2, out):
    assert(np.array_equal(in1 % in2, out))


def assert_b_equal(in1, in2, out):
    assert(np.array_equal(np.equal(in1, in2).astype(int), out))


def assert_b_not_equal(in1, in2, out):
    assert(not np.array_equal(np.equal(in1, in2).astype(int), out))


def assert_b_max(in1, in2, out):
    assert(np.array_equal(np.maximum(in1, in2), out))


def assert_b_min(in1, in2, out):
    assert(np.array_equal(np.minimum(in1, in2), out))


def assert_b_pow(in1, in2, out):
    assert(np.array_equal(np.power(in1, in2).astype(int), out))


def test_add():
    assert_b_add(*broadcast_test((5,), (1,), hlib.op.op.broadcast_add))
    assert_b_add(*broadcast_test((5, 2, 2), (1, 2, 2), hlib.op.op.broadcast_add))
    assert_b_add(*broadcast_test((2, 2, 2), (1, 2, 1), hlib.op.op.broadcast_add))
    assert_b_add(*broadcast_test((2, 2), (2, 2), hlib.op.op.broadcast_add))

def test_sub():
    assert_b_sub(*broadcast_test((5,), (1,), hlib.op.op.broadcast_sub))
    assert_b_sub(*broadcast_test((5, 2, 2), (1, 2, 2), hlib.op.op.broadcast_sub))
    assert_b_sub(*broadcast_test((2, 2, 2), (1, 2, 1), hlib.op.op.broadcast_sub))
    assert_b_sub(*broadcast_test((2, 2), (2, 2), hlib.op.op.broadcast_sub))

def test_mul():
    assert_b_mul(*broadcast_test((5,), (1,), hlib.op.op.broadcast_mul))
    assert_b_mul(*broadcast_test((5, 2, 2), (1, 2, 2), hlib.op.op.broadcast_mul))
    assert_b_mul(*broadcast_test((2, 2, 2), (1, 2, 1), hlib.op.op.broadcast_mul))
    assert_b_mul(*broadcast_test((2, 2), (2, 2), hlib.op.op.broadcast_mul))

def test_div():
    assert_b_div(*broadcast_test((5,), (1,), hlib.op.op.broadcast_div))
    assert_b_div(*broadcast_test((5, 2, 2), (1, 2, 2), hlib.op.op.broadcast_div))
    assert_b_div(*broadcast_test((2, 2, 2), (1, 2, 1), hlib.op.op.broadcast_div))
    assert_b_div(*broadcast_test((2, 2), (2, 2), hlib.op.op.broadcast_div))

def test_mod():
    assert_b_mod(*broadcast_test((5,), (1,), hlib.op.op.broadcast_mod))
    assert_b_mod(*broadcast_test((5, 2, 2), (1, 2, 2), hlib.op.op.broadcast_mod))
    assert_b_mod(*broadcast_test((2, 2, 2), (1, 2, 1), hlib.op.op.broadcast_mod))
    assert_b_mod(*broadcast_test((2, 2), (2, 2), hlib.op.op.broadcast_mod))

def test_neq():
    assert_b_not_equal(*broadcast_test((5,), (1,), hlib.op.op.broadcast_not_equal))
    assert_b_not_equal(*broadcast_test((5, 2, 2), (1, 2, 2),
                                       hlib.op.op.broadcast_not_equal))
    assert_b_not_equal(*broadcast_test((2, 2, 2), (1, 2, 1),
                                       hlib.op.op.broadcast_not_equal))
    assert_b_not_equal(*broadcast_test((2, 2), (2, 2),
                                       hlib.op.op.broadcast_not_equal))

def test_eq():
    assert_b_equal(*broadcast_test((5,), (1,), hlib.op.op.broadcast_equal))
    assert_b_equal(*broadcast_test((5, 2, 2), (1, 2, 2),
                                   hlib.op.op.broadcast_equal))
    assert_b_equal(*broadcast_test((2, 2, 2), (1, 2, 1),
                                   hlib.op.op.broadcast_equal))
    assert_b_equal(*broadcast_test((2, 2), (2, 2), hlib.op.op.broadcast_equal))

def test_max():
    assert_b_max(*broadcast_test((5,),(1,),hlib.op.op.broadcast_max))
    assert_b_max(*broadcast_test((5,2,2),(1,2,2),hlib.op.op.broadcast_max))
    assert_b_max(*broadcast_test((2,2,2),(1,2,1),hlib.op.op.broadcast_max))
    assert_b_max(*broadcast_test((2,2),(2,2),hlib.op.op.broadcast_max))

def test_min():
    assert_b_min(*broadcast_test((5,),(1,),hlib.op.op.broadcast_min))
    assert_b_min(*broadcast_test((5,2,2),(1,2,2),hlib.op.op.broadcast_min))
    assert_b_min(*broadcast_test((2,2,2),(1,2,1),hlib.op.op.broadcast_min))
    assert_b_min(*broadcast_test((2,2),(2,2),hlib.op.op.broadcast_min))

def test_pow():
    assert_b_pow(*broadcast_test((5,), (1,), hlib.op.op.broadcast_pow))
    assert_b_pow(*broadcast_test((5, 2, 2), (1, 2, 2), hlib.op.op.broadcast_pow))
    assert_b_pow(*broadcast_test((2, 2, 2), (1, 2, 1), hlib.op.op.broadcast_pow))
    assert_b_pow(*broadcast_test((2, 2), (2, 2), hlib.op.op.broadcast_pow))
