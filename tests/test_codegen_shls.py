import heterocl as hcl
from heterocl.dsl import def_

target="shls"


def test_hierarchical():

    hcl.init()

    def algo(A, B):

        @hcl.def_([A.shape, ()])
        def add_one(A, x):
            hcl.return_(A[x] + 1)

        @hcl.def_([A.shape, B.shape, ()])
        def find_max(A, B, x):
            with hcl.if_(A[x] > B[x]):
                hcl.return_(add_one(A, x))
            with hcl.else_():
                hcl.return_(B[x]) 
        
        C = hcl.compute((10,), lambda x: A[x] * B[x], "C")
        tmp = hcl.compute((10,), lambda x : find_max(A, B, x))
        D = hcl.compute((10,), lambda x: C[x] + tmp[x] + B[x], "D")
        return D

    A = hcl.placeholder((10,), "A")
    B = hcl.placeholder((10,), "B")
    s = hcl.create_schedule([A, B], algo)
    s.partition(algo.C, hcl.Partition.Block)
    code = hcl.build(s, target=target, name="dut")
    print("--------------- IR ------------------")
    print(hcl.lower(s))
    print("--------------- code ----------------")
    print(code)

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
    print(code)
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
    print(code)
    assert "for (sc_int<32> ff_outer = 0; ff_outer < 13; ++ff_outer)" in code
    assert "for (sc_int<32> ff_inner = 0; ff_inner < 5; ++ff_inner)" in code
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
    assert "((sc_int<33>)0)" in code
    assert "((sc_int<33>)(((sc_int<33>)A" in code

def test_uint_imm_ops():
    A = hcl.placeholder((10, 10), "A", dtype=hcl.UInt(1))
    def kernel(A):
        return hcl.compute((8, 8), lambda y, x:
            hcl.select(x < 4, A[y][x], 0), "B")
    s = hcl.create_scheme(A, kernel)
    s = hcl.create_schedule_from_scheme(s)
    code = hcl.build(s, target=target)
    assert "(sc_uint<32>)0U)" in code

def test_binary_ops():
    A = hcl.placeholder((8, 8), "A", dtype=hcl.Int(20))
    B = hcl.placeholder((8, 8), "B", dtype=hcl.Fixed(16,12))
    def kernel(A, B):
        return hcl.compute((8, 8), lambda y, x:
            hcl.select(x < 4, A[y][x], B[y][x]), "C", dtype=hcl.Int(8))
    s = hcl.create_scheme([A, B], kernel)
    s = hcl.create_schedule_from_scheme(s)
    code = hcl.build(s, target=target)
    assert "(sc_fixed<32, 20>)B" in code

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
    assert "sc_ufixed<20, 8>)A" in code
    print("test_uint_int() finished")

def test_easy():
    A = hcl.placeholder((10,10), "A", dtype=hcl.Int(32))
    B = hcl.placeholder((10,10), "B", dtype=hcl.Int(32))
    def kernel(A, B):
        M = hcl.compute((10,10), lambda x,y : 0, "M", dtype=hcl.Int(32))
        return hcl.compute((10,10), lambda x,y : A[x][y] + B[x][y] - M[x][y], "C", dtype=hcl.Int(32))
    s = hcl.create_schedule([A,B], kernel)
    print("-------------- IR ------------")
    print(hcl.lower(s))
    code = hcl.build(s, target=target)
    print("-------------- Generated Code ------------")
    print(code)

if __name__ == "__main__":
    # test_easy()
    # test_binary_conv()
    test_hierarchical()