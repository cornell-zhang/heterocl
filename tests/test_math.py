import heterocl as hcl
import numpy as np
import pytest

@pytest.mark.skip(reason="data type inference to be supported")
def test_exp():
    shape = (1,10)
    test_dtype = hcl.Int(32)

    def loop_body(data):
        # return hcl.compute(
        #     shape, 
        #     lambda x,y: hcl.exp(data[x,y]),
        #     name="loop_body",
        #     dtype=test_dtype,
        #  )
        return hcl.compute(shape, lambda x,y: data[x][y], "loop_body",dtype=test_dtype)


    A = hcl.placeholder(shape, "A", dtype=test_dtype)
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)

    np_a = np.random.randint(5, size=shape)
    hcl_A = hcl.asarray(np_a, dtype=test_dtype)
    hcl_B = hcl.asarray(np.zeros(shape), dtype=test_dtype)
    f(hcl_A, hcl_B)
    np_b = hcl_B.asnumpy()
    b_golden = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            b_golden[i,j] = np.exp(np_a[i,j])

    print(b_golden)
    print(np_b)
    assert np.allclose(np_b, b_golden)

def test_power():
    shape = (1,10)

    def loop_body(data, power):
        return hcl.compute(shape, lambda x,y: hcl.power(data[x,y],power[x,y]), "loop_body")

    A = hcl.placeholder(shape, "A")
    B = hcl.placeholder(shape, "B")
    s = hcl.create_schedule([A, B], loop_body)
    f = hcl.build(s)

    np_a = np.random.randint(5, size=shape)
    np_b = np.random.randint(5, size=shape)
    hcl_A = hcl.asarray(np_a)
    hcl_B = hcl.asarray(np_b)
    hcl_C = hcl.asarray(np.zeros(shape))
    f(hcl_A, hcl_B, hcl_C)
    np_c = hcl_C.asnumpy()
    c_golden = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            c_golden[i,j] = np.power(np_a[i,j],np_b[i,j])
    assert np.allclose(np_c, c_golden)

@pytest.mark.skip(reason="data type inference to be supported")
def test_log():
    shape = (1,10)

    def loop_body(data):
        # return hcl.compute(shape, lambda x,y: hcl.log(data[x,y]), "loop_body")
        return hcl.compute(shape, lambda x,y: data[x][y], "loop_body")

    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)

    np_a = np.random.randint(5, size=shape)
    hcl_A = hcl.asarray(np_a)
    hcl_B = hcl.asarray(np.zeros(shape))
    f(hcl_A, hcl_B)
    np_b = hcl_B.asnumpy()
    b_golden = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            b_golden[i,j] = np.log(np_a[i,j])
    print(b_golden)
    print(np_b)
    assert np.allclose(np_b, b_golden)

@pytest.mark.skip(reason="data type inference to be supported")
def test_log2():
    shape = (1,10)

    def loop_body(data):
        # return hcl.compute(shape, lambda x,y: hcl.log2(data[x,y]), "loop_body")
        return hcl.compute(shape, lambda x,y: data[x][y], "loop_body")


    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)

    np_a = np.random.randint(5, size=shape)
    hcl_A = hcl.asarray(np_a)
    hcl_B = hcl.asarray(np.zeros(shape))
    f(hcl_A, hcl_B)
    np_b = hcl_B.asnumpy()
    b_golden = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            b_golden[i,j] = np.log2(np_a[i,j])
    print(b_golden)
    print(np_b)
    assert np.allclose(np_b, b_golden)

@pytest.mark.skip(reason="data type inference to be supported")
def test_log10():
    shape = (1,10)

    def loop_body(data):
        # return hcl.compute(shape, lambda x,y: hcl.log10(data[x,y]), "loop_body")
        return hcl.compute(shape, lambda x,y: data[x][y], "loop_body")

    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)

    np_a = np.random.randint(5, size=shape)
    hcl_A = hcl.asarray(np_a)
    hcl_B = hcl.asarray(np.zeros(shape))
    f(hcl_A, hcl_B)
    np_b = hcl_B.asnumpy()
    b_golden = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            b_golden[i,j] = np.log10(np_a[i,j])
    print(b_golden)
    print(np_b)
    assert np.allclose(np_b, b_golden)

@pytest.mark.skip(reason="data type inference to be supported")
def test_sqrt():
    shape = (1,10)

    def loop_body(data):
        # return hcl.compute(shape, lambda x,y: hcl.sqrt(data[x,y]), "loop_body")
        return hcl.compute(shape, lambda x,y: data[x][y], "loop_body")


    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)

    np_a = np.random.randint(100, size=shape)
    hcl_A = hcl.asarray(np_a)
    hcl_B = hcl.asarray(np.zeros(shape))
    f(hcl_A, hcl_B)
    np_b = hcl_B.asnumpy()
    b_golden = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            b_golden[i,j] = np.sqrt(np_a[i,j])
    print(b_golden)
    print(np_b)
    assert np.allclose(np_b, b_golden)

@pytest.mark.skip(reason="data type inference to be supported")
def test_sin():
    shape = (1,10)

    def loop_body(data):
        # return hcl.compute(shape, lambda x,y: hcl.sin(data[x,y]), "loop_body")
        return hcl.compute(shape, lambda x,y: data[x][y], "loop_body")

    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)

    np_a = np.random.randint(10, size=shape)
    hcl_A = hcl.asarray(np_a)
    hcl_B = hcl.asarray(np.zeros(shape))
    f(hcl_A, hcl_B)
    np_b = hcl_B.asnumpy()
    b_golden = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            b_golden[i,j] = np.sin(np_a[i,j])
    print(b_golden)
    print(np_b)
    assert np.allclose(np_b, b_golden)

@pytest.mark.skip(reason="data type inference to be supported")
def test_cos():
    shape = (1,10)

    def loop_body(data):
        # return hcl.compute(shape, lambda x,y: hcl.cos(data[x,y]), "loop_body")
        return hcl.compute(shape, lambda x,y: data[x][y], "loop_body")

    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)

    np_a = np.random.randint(10, size=shape)
    hcl_A = hcl.asarray(np_a)
    hcl_B = hcl.asarray(np.zeros(shape))
    f(hcl_A, hcl_B)
    np_b = hcl_B.asnumpy()
    b_golden = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            b_golden[i,j] = np.cos(np_a[i,j])
    print(b_golden)
    print(np_b)
    assert np.allclose(np_b, b_golden)

@pytest.mark.skip(reason="data type inference to be supported")
def test_tanh():
    shape = (1,10)

    def loop_body(data):
        return hcl.compute(shape, lambda x,y: hcl.tanh(data[x,y]), "loop_body")
        # return hcl.compute(shape, lambda x,y: data[x][y], "loop_body")

    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)

    np_a = np.random.randint(10, size=shape)
    hcl_A = hcl.asarray(np_a)
    hcl_B = hcl.asarray(np.zeros(shape))
    f(hcl_A, hcl_B)
    np_b = hcl_B.asnumpy()
    b_golden = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            b_golden[i,j] = np.tanh(np_a[i,j])
    print(b_golden)
    print(np_b)
    assert np.allclose(np_b, b_golden)
    