import heterocl as hcl

target="shls"

def test_pack():

    def pack(A):
        return hcl.pack(A, factor=5)

    A = hcl.placeholder((40,), "A", dtype = hcl.UInt(3))
    s = hcl.create_schedule([A], pack)
    code = hcl.build(s, target=target)
    slice_range = "(((i * 3) + 2), (i * 3))"
    assert slice_range in code

def test_index_split():
    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    B = hcl.compute(A.shape, lambda y, x: A[y][x], "B")
    s = hcl.create_schedule([A, B])
    s[B].split(B.axis[0], 5)
    code = hcl.build(s, target=target)
    print(code)
    assert "B[(y_inner + (y_outer * 5))][x]" in code

def test_index_split_reshape():
    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    B = hcl.compute(A.shape, lambda y, x: A[y][x], "B")
    s = hcl.create_schedule([A, B])
    s[B].split(B.axis[0], 5)
    s.reshape(B, (2, 5, 10))
    code = hcl.build(s, target=target)
    assert "B[y_outer][y_inner][x]" in code

def test_index_fuse():
    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    B = hcl.compute(A.shape, lambda y, x: A[y][x], "B")
    s = hcl.create_schedule([A, B])
    s[B].fuse(B.axis[0], B.axis[1])
    code = hcl.build(s, target=target)
    assert "B[(y_x_fused / 10)][(y_x_fused % 10)]" in code

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
    code = hcl.build(s, target=target)
    assert "for (ap_int<32> ff_outer = 0; ff_outer < 13; ++ff_outer)" in code
    assert "for (ap_int<32> ff_inner = 0; ff_inner < 5; ++ff_inner)" in code
    assert "if (ff_inner < (64 - (ff_outer * 5)))" in code

def test_legacy_interface():
    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    B = hcl.compute(A.shape, lambda y, x: A[y][x], "B")
    s = hcl.create_schedule([A, B])
    s[B].fuse(B.axis[0], B.axis[1])
    code = hcl.build(s, target=target)
    assert "A[10][10]" in code
    assert "B[10][10]" in code

def test_imm_ops():
    A = hcl.placeholder((10, 10), "A")
    def kernel(A):
        return hcl.compute((8, 8), lambda y, x:
            hcl.select(x < 4, A[y][x] + A[y+2][x+2], 0), "B")
    s = hcl.create_scheme(A, kernel)
    s = hcl.create_schedule_from_scheme(s)
    code = hcl.build(s, target=target)
    assert "((ap_int<33>)0)" in code
    assert "((ap_int<33>)(((ap_int<33>)A" in code

def test_uint_imm_ops():
    A = hcl.placeholder((10, 10), "A", dtype=hcl.UInt(1))
    def kernel(A):
        return hcl.compute((8, 8), lambda y, x:
            hcl.select(x < 4, A[y][x], 0), "B")
    s = hcl.create_scheme(A, kernel)
    s = hcl.create_schedule_from_scheme(s)
    code = hcl.build(s, target=target)
    assert "(ap_uint<32>)0U)" in code

def test_binary_ops():
    A = hcl.placeholder((8, 8), "A", dtype=hcl.Int(20))
    B = hcl.placeholder((8, 8), "B", dtype=hcl.Fixed(16,12))
    def kernel(A, B):
        return hcl.compute((8, 8), lambda y, x:
            hcl.select(x < 4, A[y][x], B[y][x]), "C", dtype=hcl.Int(8))
    s = hcl.create_scheme([A, B], kernel)
    s = hcl.create_schedule_from_scheme(s)
    code = hcl.build(s, target=target)
    assert "(ap_fixed<32, 20>)B" in code

def test_uint_int():
    A = hcl.placeholder((8, 8), "A", dtype=hcl.Fixed(20,12))
    B = hcl.placeholder((8, 8), "B", dtype=hcl.UFixed(16,12))
    def kernel(A, B):
        return hcl.compute((8, 8), lambda y, x:
            hcl.select(x < 4, A[y][x], B[y][x]), "C", dtype=hcl.Int(8))
    s = hcl.create_scheme([A, B], kernel)
    s = hcl.create_schedule_from_scheme(s)
    code = hcl.build(s, target=target)
    print(code)
    assert "ap_ufixed<20, 8>)A" in code
    print("test_uint_int() finished")

def test_easy():
    A = hcl.placeholder((10,), "A", dtype=hcl.Int(32))
    B = hcl.placeholder((10,), "B", dtype=hcl.Int(32))
    def kernel(A, B):
        return hcl.compute((10,), lambda x : A[x] + B[x], "C", dtype=hcl.Int(32))
    s = hcl.create_schedule([A,B], kernel)
    print("-------------- IR ------------")
    print(hcl.lower(s))
    code = hcl.build(s, target=target)
    print("-------------- Generated Code ------------")
    print(code)

if __name__ == "__main__":
    test_easy()
