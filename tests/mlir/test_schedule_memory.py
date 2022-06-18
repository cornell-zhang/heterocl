import heterocl as hcl
import numpy as np
import pytest

def test_reuse_blur_x():
    hcl.init()
    A = hcl.placeholder((10, 10))
    B = hcl.compute((10, 8), lambda y, x: A[y, x] + A[y, x+1] + A[y, x+2])
    s = hcl.create_schedule([A])
    RB = s.reuse_at(A, s[B], B.axis[1])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((10, 8), dtype="int")
    np_C = np.zeros((10, 8), dtype="int")

    for y in range(0, 10):
        for x in range(0, 8):
            np_C[y][x] = np_A[y][x] + np_A[y][x+1] + np_A[y][x+2]

    hcl_A = hcl.asarray(np_A, dtype=hcl.Int(32))
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_reuse_blur_x_tensor():
    hcl.init()
    A = hcl.placeholder((10, 10))
    X = hcl.compute((10, 10), lambda y, x: A[y, x])
    B = hcl.compute((10, 8), lambda y, x: X[y, x] + X[y, x+1] + X[y, x+2])
    s = hcl.create_schedule([A])
    RB = s.reuse_at(X, s[B], B.axis[1])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((10, 8), dtype="int")
    np_C = np.zeros((10, 8), dtype="int")

    for y in range(0, 10):
        for x in range(0, 8):
            np_C[y][x] = np_A[y][x] + np_A[y][x+1] + np_A[y][x+2]

    hcl_A = hcl.asarray(np_A, dtype=hcl.Int(32))
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_reuse_blur_y():
    hcl.init()
    A = hcl.placeholder((10, 10))
    B = hcl.compute((8, 10), lambda y, x: A[y, x] + A[y+1, x] + A[y+2, x])
    s = hcl.create_schedule([A])
    RB = s.reuse_at(A, s[B], B.axis[0])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((8, 10), dtype="int")
    np_C = np.zeros((8, 10), dtype="int")

    for y in range(0, 8):
        for x in range(0, 10):
            np_C[y][x] = np_A[y][x] + np_A[y+1][x] + np_A[y+2][x]

    hcl_A = hcl.asarray(np_A, dtype=hcl.Int(32))
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_reuse_blur_x_y():
    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    B = hcl.compute((8, 8), lambda y, x: A[y, x] + A[y+1, x+1] + A[y+2, x+2], "B")
    s = hcl.create_schedule([A])
    RB_y = s.reuse_at(A, s[B], B.axis[0], "RB_y")
    RB_x = s.reuse_at(RB_y, s[B], B.axis[1], "RB_x")
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((8, 8), dtype="int")
    np_C = np.zeros((8, 8), dtype="int")

    for y in range(0, 8):
        for x in range(0, 8):
            np_C[y][x] = np_A[y][x] + np_A[y+1][x+1] + np_A[y+2][x+2]

    hcl_A = hcl.asarray(np_A, dtype=hcl.Int(32))
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_reuse_blur_x_3D():
    hcl.init()
    A = hcl.placeholder((10, 10, 2))
    B = hcl.compute((10, 8, 2), lambda y, x, c: A[y, x, c] + A[y, x+1, c] + A[y, x+2, c])
    s = hcl.create_schedule([A])
    RB = s.reuse_at(A, s[B], B.axis[1])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10, 2))
    np_B = np.zeros((10, 8, 2), dtype="int")
    np_C = np.zeros((10, 8, 2), dtype="int")

    for y in range(0, 10):
        for x in range(0, 8):
            for c in range(0, 2):
                np_C[y][x][c] = np_A[y][x][c] + np_A[y][x+1][c] + np_A[y][x+2][c]

    hcl_A = hcl.asarray(np_A, hcl.Int(32))
    hcl_B = hcl.asarray(np_B, hcl.Int(32))

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_reuse_blur_y_3D():
    hcl.init()
    A = hcl.placeholder((10, 10, 2))
    B = hcl.compute((8, 10, 2), lambda y, x, c: A[y, x, c] + A[y+1, x, c] + A[y+2, x, c])
    s = hcl.create_schedule([A])
    RB = s.reuse_at(A, s[B], B.axis[0])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10, 2))
    np_B = np.zeros((8, 10, 2), dtype="int")
    np_C = np.zeros((8, 10, 2), dtype="int")

    for y in range(0, 8):
        for x in range(0, 10):
            for c in range(0, 2):
                np_C[y][x][c] = np_A[y][x][c] + np_A[y+1][x][c] + np_A[y+2][x][c]

    hcl_A = hcl.asarray(np_A, hcl.Int(32))
    hcl_B = hcl.asarray(np_B, hcl.Int(32))

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_reuse_blur_x_y_3D():
    hcl.init()
    A = hcl.placeholder((10, 10, 2), "A")
    B = hcl.compute((8, 8, 2), lambda y, x, c: A[y, x, c] + A[y+1, x+1, c] + A[y+2, x+2, c], "B")
    s = hcl.create_schedule([A])
    RB_y = s.reuse_at(A, s[B], B.axis[0], "RB_y")
    RB_x = s.reuse_at(RB_y, s[B], B.axis[1], "RB_x")
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10, 2))
    np_B = np.zeros((8, 8, 2), dtype="int")
    np_C = np.zeros((8, 8, 2), dtype="int")

    for y in range(0, 8):
        for x in range(0, 8):
            for c in range(0, 2):
                np_C[y][x][c] = np_A[y][x][c] + np_A[y+1][x+1][c] + np_A[y+2][x+2][c]

    hcl_A = hcl.asarray(np_A, hcl.Int(32))
    hcl_B = hcl.asarray(np_B, hcl.Int(32))

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_reuse_blur_x_y_z_3D():
    hcl.init()
    A = hcl.placeholder((10, 8, 6), "A")
    B = hcl.compute((8, 6, 4), lambda y, x, z: A[y, x, z] + A[y+1, x+1, z+1] + A[y+2, x+2, z+2], "B")
    s = hcl.create_schedule([A])
    RB_y = s.reuse_at(A, s[B], B.axis[0], "RB_y")
    RB_x = s.reuse_at(RB_y, s[B], B.axis[1], "RB_x")
    RB_z = s.reuse_at(RB_x, s[B], B.axis[2], "RB_z")
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 8, 6))
    np_B = np.zeros((8, 6, 4), dtype="int")
    np_C = np.zeros((8, 6, 4), dtype="int")

    for y in range(0, 8):
        for x in range(0, 6):
            for z in range(0, 4):
                np_C[y][x][z] = np_A[y][x][z] + np_A[y+1][x+1][z+1] + np_A[y+2][x+2][z+2]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_conv2D_lb():
    hcl.init()
    A = hcl.placeholder((10, 10))
    r = hcl.reduce_axis(0, 3)
    c = hcl.reduce_axis(0, 3)
    B = hcl.compute((8, 8), lambda y, x: hcl.sum(A[y+r, x+c], axis=[r, c]))
    s = hcl.create_schedule([A])
    LB = s.reuse_at(A, s[B], B.axis[0])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((8, 8), dtype="int")
    np_C = np.zeros((8, 8), dtype="int")

    for y in range(0, 8):
        for x in range(0, 8):
            for r in range(0, 3):
                for c in range(0, 3):
                    np_C[y][x] += np_A[y+r][x+c]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_conv2D_wb():
    hcl.init()
    A = hcl.placeholder((10, 10))
    r = hcl.reduce_axis(0, 3)
    c = hcl.reduce_axis(0, 3)
    B = hcl.compute((8, 8), lambda y, x: hcl.sum(A[y+r, x+c], axis=[r, c]))
    s = hcl.create_schedule([A])
    WB = s.reuse_at(A, s[B], B.axis[1])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((8, 8), dtype="int")
    np_C = np.zeros((8, 8), dtype="int")

    for y in range(0, 8):
        for x in range(0, 8):
            for r in range(0, 3):
                for c in range(0, 3):
                    np_C[y][x] += np_A[y+r][x+c]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_conv2D_lb_wb():
    hcl.init()
    A = hcl.placeholder((10, 10))
    r = hcl.reduce_axis(0, 3)
    c = hcl.reduce_axis(0, 3)
    B = hcl.compute((8, 8), lambda y, x: hcl.sum(A[y+r, x+c], axis=[r, c]))
    s = hcl.create_schedule([A])
    LB = s.reuse_at(A, s[B], B.axis[0])
    WB = s.reuse_at(LB, s[B], B.axis[1])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((8, 8), dtype="int")
    np_C = np.zeros((8, 8), dtype="int")

    for y in range(0, 8):
        for x in range(0, 8):
            for r in range(0, 3):
                for c in range(0, 3):
                    np_C[y][x] += np_A[y+r][x+c]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

def test_partition_basic():
    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    B = hcl.compute(A.shape, lambda x, y: A[x, y], "B")
    s = hcl.create_schedule([A])
    s.partition(A)
    ir = str(hcl.lower(s))
    assert "affine_map<(d0, d1) -> (d0, d1, 0, 0)>" in ir

def test_partition_type():
    def test_1():
        A = hcl.placeholder((10, 10), "A")
        B = hcl.compute(A.shape, lambda x, y: A[x, y], "B")
        s1 = hcl.create_schedule([A])
        s1.partition(A)
        ir = str(hcl.lower(s1))
        assert "affine_map<(d0, d1) -> (d0, d1, 0, 0)>" in ir
    
    def test_2():
        A = hcl.placeholder((10, 10), "A")
        B = hcl.compute(A.shape, lambda x, y: A[x, y], "B")
        s1 = hcl.create_schedule([A])
        s1.partition(A, hcl.Partition.Block, dim=0, factor=2)
        ir = str(hcl.lower(s1))
        assert "affine_map<(d0, d1) -> (d0 floordiv 5, d1 floordiv 5, d0 mod 5, d1 mod 5)>" in ir
    
    def test_3():
        A = hcl.placeholder((10, 10), "A")
        B = hcl.compute(A.shape, lambda x, y: A[x, y], "B")
        s1 = hcl.create_schedule([A])
        s1.partition(A, hcl.Partition.Cyclic)
        ir = str(hcl.lower(s1))
        assert "affine_map<(d0, d1) -> (d0 mod 0, d1 mod 0, d0 floordiv 0, d1 floordiv 0)>" in ir

    test_1()
    test_2()
    test_3()

def test_partition_dim_factor():
    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    B = hcl.compute(A.shape, lambda x, y: A[x, y], "B")
    s = hcl.create_schedule([A])
    s.partition(A, dim=1, factor=2)
    ir = str(hcl.lower(s))
    assert "affine_map<(d0, d1) -> (d0, 0, 0, d1)>" in ir

def test_reshape():
    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    B = hcl.compute(A.shape, lambda x, y: A[x, y], "B")
    C = hcl.compute(A.shape, lambda x, y: B[x, y], "C")
    s = hcl.create_schedule([A])
    s.reshape(B, (2, 5, 2, 5))
    ir = str(hcl.lower(s))
    assert "memref<2x5x2x5xi32>" in ir

@pytest.mark.skip(reason="crashed")
def test_conv2D_lb_wb_schedule():
    hcl.init()
    A = hcl.placeholder((10, 10))
    r = hcl.reduce_axis(0, 3)
    c = hcl.reduce_axis(0, 3)
    B = hcl.compute((8, 8), lambda y, x: hcl.sum(A[y+r, x+c], axis=[r, c]))
    s = hcl.create_schedule([A])
    xo, xi = s[B].split(B.axis[1], 4)
    s[B].reorder(xo, B.axis[0], xi)
    LB = s.reuse_at(A, s[B], B.axis[0])
    WB = s.reuse_at(LB, s[B], xi)
    s.partition(LB, dim=2)
    s.partition(WB)
    s.reshape(B, (8, 2, 4))
    s[B].pipeline(B.axis[0])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((8, 2, 4), dtype="int")
    np_C = np.zeros((8, 2, 4), dtype="int")

    for y in range(0, 8):
        for xo in range(0, 2):
            for xi in range(0, 4):
                for r in range(0, 3):
                    for c in range(0, 3):
                        np_C[y][xo][xi] += np_A[y+r][xi+xo*4+c]

    hcl_A = hcl.asarray(np_A, dtype=hcl.Int(32))
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)
