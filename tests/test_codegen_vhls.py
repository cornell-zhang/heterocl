import heterocl as hcl

def test_dtype():
    def test_ap_int():
        hcl.init()
        A = hcl.placeholder((1, 32), dtype=hcl.Int(3))
        B = hcl.placeholder((1, 32), dtype=hcl.UInt(3))
        C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], dtype=hcl.Int(8))
        s = hcl.create_schedule([A, B, C])
        code = hcl.build(s, target='vhls')
        assert "ap_int<3>" in code
        assert "ap_uint<3>" in code
        assert "ap_int<8>" in code

    def test_ap_fixed():
        hcl.init()
        A = hcl.placeholder((1, 32), dtype=hcl.Fixed(5, 3))
        B = hcl.placeholder((1, 32), dtype=hcl.UFixed(5, 3))
        C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], dtype=hcl.Fixed(7, 4))
        s = hcl.create_schedule([A, B, C])
        code = hcl.build(s, target='vhls')
        assert "ap_fixed<5, 2>" in code
        assert "ap_ufixed<5, 2>" in code
        assert "ap_fixed<7, 3>" in code

    test_ap_int()
    test_ap_fixed()

def test_pragma():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32))
    C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j])
    # unroll
    s1 = hcl.create_schedule([A, B, C])
    s1[C].unroll(C.axis[1], factor=4)
    code1 = hcl.build(s1, target='vhls')
    assert "#pragma HLS unroll factor=4" in code1
    # pipeline
    s2 = hcl.create_schedule([A, B, C])
    s2[C].pipeline(C.axis[0], initiation_interval=2)
    code2 = hcl.build(s2, target='vhls')
    assert "#pragma HLS pipeline II=2" in code2
    # partition
    s3 = hcl.create_schedule([A, B, C])
    s3.partition(A, hcl.Partition.Block, dim=2, factor=2)
    code3 = hcl.build(s3, target='vhls')
    assert "#pragma HLS array_partition variable=A block dim=2 factor=2" in code3

def test_set_bit():

    A = hcl.placeholder((10,), "A")
    def kernel(A):
        with hcl.Stage("S"):
            A[0][4] = 1
    s = hcl.create_schedule([A], kernel)
    code = hcl.build(s, target="vhls")
    assert "A[0][4] = 1" in code

def test_set_slice():

    A = hcl.placeholder((10,), "A")
    def kernel(A):
        with hcl.Stage("S"):
            A[0][5:1] = 1
    s = hcl.create_schedule([A], kernel)
    code = hcl.build(s, target="vhls")
    assert "A[0](4, 1) = 1" in code

def test_pack():

    def pack(A):
        return hcl.pack(A, factor=5)

    A = hcl.placeholder((40,), "A", dtype = hcl.UInt(3))
    s = hcl.create_schedule([A], pack)
    code = hcl.build(s, target="vhls")
    slice_range = "(((i * 3) + 2), (i * 3))"
    assert slice_range in code


def test_binary_conv():
    hcl.init()
    A = hcl.placeholder((1, 32, 14, 14), dtype=hcl.UInt(1), name="A")
    B = hcl.placeholder((64, 32, 3, 3), dtype=hcl.UInt(1), name="B")
    rc = hcl.reduce_axis(0, 32)
    ry = hcl.reduce_axis(0, 3)
    rx = hcl.reduce_axis(0, 3)
    C = hcl.compute((1, 64, 12, 12),
        lambda nn, ff, yy, xx: hcl.sum(
            A[nn, rc, yy + ry, xx + rx] * B[ff, rc, ry, rx], axis=[rc, ry, rx]),
        dtype=hcl.UInt(8), name="C")
    s = hcl.create_schedule([A, B, C])
    s[C].split(C.axis[1], factor=5)
    code = hcl.build(s, target='vhls')
    assert "for (ap_int<32> ff_outer = 0; ff_outer < 13; ++ff_outer)" in code
    assert "for (ap_int<32> ff_inner = 0; ff_inner < 5; ++ff_inner)" in code
    assert "if ((ff_outer * 5) < (64 - ff_inner))" in code


if __name__ == '__main__':
    test_dtype()
    # test_pragma()
    test_pack()
    # test_binary_conv()
