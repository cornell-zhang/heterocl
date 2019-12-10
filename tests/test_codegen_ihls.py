import heterocl as hcl

def test_ac_int():
    hcl.init()
    A = hcl.placeholder((1, 32), dtype=hcl.Int(3))
    B = hcl.placeholder((1, 32), dtype=hcl.UInt(3))
    C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], dtype=hcl.Int(8))
    s = hcl.create_schedule([A, B, C])
    code = hcl.build(s, target='ihls')
    assert "ac_int<3, true>" in code
    assert "ac_int<3, false>" in code
    assert "ac_int<8, true>" in code

def test_ac_fixed():
    hcl.init()
    A = hcl.placeholder((1, 32), dtype=hcl.Fixed(5, 3))
    B = hcl.placeholder((1, 32), dtype=hcl.UFixed(5, 3))
    C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], dtype=hcl.Fixed(7, 4))
    s = hcl.create_schedule([A, B, C])
    code = hcl.build(s, target='ihls')
    assert "ac_fixed<5, 2, true>" in code
    assert "ac_fixed<5, 2, false>" in code
    assert "ac_fixed<7, 3, true>" in code

def test_pragma():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32))
    C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j])
    # unroll
    s1 = hcl.create_schedule([A, B, C])
    s1[C].unroll(C.axis[1], factor=4)
    code1 = hcl.build(s1, target='ihls')
    assert "#pragma unroll 4" in code1
    # pipeline
    s2 = hcl.create_schedule([A, B, C])
    s2[C].pipeline(C.axis[0], initiation_interval=2)
    code2 = hcl.build(s2, target='ihls')
    assert "#pragma ii 2" in code2

def test_set_bit():
    A = hcl.placeholder((10,), "A")
    def kernel(A):
        with hcl.Stage("S"):
            A[0][4] = 1
    s = hcl.create_schedule([A], kernel)
    code = hcl.build(s, target="ihls")
    assert "A[0][4] = 1" in code

def test_set_slice():
    A = hcl.placeholder((10,), "A")
    def kernel(A):
        with hcl.Stage("S"):
            A[0][5:1] = 1
    s = hcl.create_schedule([A], kernel)
    code = hcl.build(s, target="ihls")
    assert "A[0].set_slc(1, ((ac_int<4, false>)1))" in code

def test_get_slice():

    A = hcl.placeholder((10,), "A")
    def kernel(A):
        with hcl.Stage("S"):
            A[0] = A[0][5:1]
    s = hcl.create_schedule([A], kernel)
    code = hcl.build(s, target="ihls")
    assert "A[0].slc<4>(1)" in code

