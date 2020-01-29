import heterocl as hcl
import numpy as np

def test_int():
    hcl.init(hcl.Int())

    a = hcl.placeholder((10,))
    b = hcl.compute(a.shape, lambda x: hcl.power(2, a[x]))

    s = hcl.create_schedule([a, b])
    f = hcl.build(s)

    np_a = np.random.randint(1, 10, (10,))
    np_b = np.zeros(10, dtype="int")

    hcl_a = hcl.asarray(np_a)
    hcl_b = hcl.asarray(np_b)

    f(hcl_a, hcl_b)

    np_golden = np.power(2, np_a)
    assert np.allclose(np_golden, hcl_b.asnumpy())

def test_large_int():
    hcl.init(hcl.Int())

    a = hcl.placeholder((10,))
    b = hcl.placeholder((10,))
    c = hcl.compute(a.shape, lambda x: hcl.power(a[x], b[x]))

    s = hcl.create_schedule([a, b, c])
    f = hcl.build(s)

    np_a = np.random.randint(1, 10, (10,))
    np_b = np.random.randint(1, 10, (10,))
    np_c = np.zeros(10, dtype="int")

    hcl_a = hcl.asarray(np_a)
    hcl_b = hcl.asarray(np_b)
    hcl_c = hcl.asarray(np_c)

    f(hcl_a, hcl_b, hcl_c)

    np_golden = np.power(np_a, np_b)
    assert np.allclose(np_golden, hcl_c.asnumpy())

def test_float():
    hcl.init(hcl.Float())

    a = hcl.placeholder((10,))
    b = hcl.compute(a.shape, lambda x: hcl.power(2.0, a[x]))

    s = hcl.create_schedule([a, b])
    f = hcl.build(s)

    np_a = np.random.rand(10)
    np_b = np.zeros(10)

    hcl_a = hcl.asarray(np_a)
    hcl_b = hcl.asarray(np_b)

    f(hcl_a, hcl_b)

    np_golden = np.power(2, np_a)
    assert np.allclose(np_golden, hcl_b.asnumpy())

def test_var():
    hcl.init(hcl.Float())

    a = hcl.placeholder((10,))
    b = hcl.placeholder((10,))
    c = hcl.compute(a.shape, lambda x: hcl.power(a[x], b[x]))

    s = hcl.create_schedule([a, b, c])
    f = hcl.build(s)

    np_a = np.random.rand(10)
    np_b = np.random.rand(10)
    np_c = np.zeros(10)

    hcl_a = hcl.asarray(np_a)
    hcl_b = hcl.asarray(np_b)
    hcl_c = hcl.asarray(np_c)

    f(hcl_a, hcl_b, hcl_c)

    np_golden = np.power(np_a, np_b)
    assert np.allclose(np_golden, hcl_c.asnumpy())
