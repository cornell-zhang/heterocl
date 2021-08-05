import heterocl as hcl
import numpy as np

def test_1D_basic():

    hcl.init()

    def kernel(A):
        matrix_C = A[2:7]
        return hcl.compute((5,), lambda x: matrix_C[x])

    A = hcl.placeholder((10,))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(5)
    golden = np_A[2:7]
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_1D_copy():

    hcl.init()

    def kernel(A):
        matrix_C = hcl.copy(A[2:7])
        A[2:7][0] = 123
        hcl.assert_(A[2] == 123)
        return hcl.compute((5,), lambda x: matrix_C[x])

    A = hcl.placeholder((10,))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(5)
    golden = np_A[2:7]
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_1D_broadcast():

    hcl.init()

    def kernel(A):
        A[2:7] = 999
        return hcl.compute((10,), lambda x: A[x])

    A = hcl.placeholder((10,))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(10)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)
    golden = np_A
    for x in range(2, 7):
        golden[x] = 999
    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_1D_slice():

    hcl.init()

    def kernel(A):
        matrix_C = A[7][3][2:6]
        return hcl.compute((4,), lambda x: matrix_C[x])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros(4)
    golden = np_A[7][3][2:6]
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_2D_slice():

    hcl.init()

    def kernel(A):
        matrix_C = A[7][2:6]
        return hcl.compute((4,8), lambda x, y: matrix_C[x][y])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros((4, 8))
    golden = np_A[7][2:6]
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_3D_slice():

    hcl.init()

    def kernel(A):
        matrix_C = A[2:6]
        return hcl.compute((4, 9, 8), lambda x, y, z: matrix_C[x][y][z])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros((4, 9, 8))
    golden = np_A[2:6]
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_3D_copy_dim2():

    hcl.init()

    def kernel(A):
        matrix_C = hcl.copy(A[8][1][2:7])
        A[8][1][2:7][0] = 123
        hcl.assert_(A[8][1][2] == 123)
        return hcl.compute((5,), lambda x: matrix_C[x])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros(5)
    golden = np_A[8][1][2:7]
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_3D_copy_dim1_setslice():

    hcl.init()

    def kernel(A):
        matrix_C = hcl.copy(A[8][2:7])
        A[8][3][3:8] = A[2][6][1:6]
        with hcl.for_(0, 5) as i:
            hcl.assert_(A[8][3][i + 3] == A[2][6][i + 1])
        return hcl.compute((5,8), lambda x, y: matrix_C[x][y])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros((5, 8))
    golden = np_A[8][2:7]
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_3D_copy_dim0_broadcast():

    hcl.init()

    def kernel(A):
        matrix_C = hcl.copy(A[2:7])
        A[1:4] = 999
        with hcl.for_(0, 3) as x:
            with hcl.for_(0, 9) as y:
                with hcl.for_(0, 8) as z:
                    hcl.assert_(A[x + 1][y][z] == 999)
        return hcl.compute((5, 9, 8), lambda x, y, z: matrix_C[x][y][z])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros((5, 9, 8))
    golden = np_A[2:7]
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_3D_copy_dim0():

    hcl.init()

    def kernel(A):
        matrix_C = hcl.copy(A[0])
        A[1:4] = 999
        with hcl.for_(0, 3) as x:
            with hcl.for_(0, 9) as y:
                with hcl.for_(0, 8) as z:
                    hcl.assert_(A[x + 1][y][z] == 999)
        return hcl.compute((9, 8), lambda x, y: matrix_C[x][y])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros((9, 8))
    golden = np_A[0]
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_5D_broadcast():

    hcl.init()

    def kernel(A):
        A[1:3] = 999
        return hcl.compute((4, 3, 2, 3, 2), lambda x, y, z, a, b: A[x][y][z][a][b])

    A = hcl.placeholder((4, 3, 2, 3, 2))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(4, 3, 2, 3, 2))
    np_B = np.zeros((4, 3, 2, 3, 2))
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)
    golden = np_A
    for x in range(1, 3):
        for y in range(0, 3):
            for z in range(0, 2):
                for a in range(0, 3):
                    for b in range(0, 2):
                        golden[x][y][z][a][b] = 999

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_nested_slice_dim1():

    hcl.init()

    def kernel(A):
        return hcl.compute((10, 3, 8), lambda x, y, z: A[x][2:9][3:6][y][z])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros((10, 3, 8))
    golden = np.zeros((10, 3, 8))
    for x in range(0, 10):
        for y in range(0, 3):
            for z in range(0, 8):
                golden[x][y][z] = np_A[x][y+5][z]
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)
    golden = np.zeros((10, 3, 8))
    for x in range(0, 10):
        for y in range(0, 3):
            for z in range(0, 8):
                golden[x][y][z] = np_A[x][y+5][z]

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_nested_slice_dim2():

    hcl.init()

    def kernel(A):
        return hcl.compute((10, 9, 3), lambda x, y, z: A[x][y][2:8][3:6][z])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros((10, 9, 3))
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)
    golden = np.zeros((10, 9, 3))
    for x in range(0, 10):
        for y in range(0, 9):
            for z in range(0, 3):
                golden[x][y][z] = np_A[x][y][z+5]

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_nested_slice_dim0():

    hcl.init()

    def kernel(A):
        return hcl.compute((3, 9, 8), lambda x, y, z: A[2:8][3:6][x][y][z])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros((3, 9, 8))
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)
    golden = np.zeros((3, 9, 8))
    for x in range(0, 3):
        for y in range(0, 9):
            for z in range(0, 8):
                golden[x][y][z] = np_A[x+5][y][z]

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_double_slice():

    hcl.init()

    def kernel(A):
        matrix_B = A[1][2:7][3:5]
        return hcl.compute((2, 8), lambda x, y: matrix_B[x][y])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros((2, 8))
    golden = np_A[1][2:7][3:5]
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_get_bitslice_1D_tensor():

    hcl.init()

    def kernel(A):
        matrix_B = A[3:7]
        return hcl.compute((4,), lambda x: matrix_B[x][2:0])

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(4)
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    golden = np_A[3:7] & 0b11
    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_get_bitslice_3D_tensorslice():

    hcl.init()

    def kernel(A):
        return hcl.compute((4, 9, 8), lambda x, y, z: A[3:7][x][y][z][3:1])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros((4, 9, 8))
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    golden = (np_A[3:7] & 0b110) >> 1
    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_get_bitslice_1D_tensorslice():

    hcl.init()

    def kernel(A):
        matrix_B = A[6][1][2:7]
        return hcl.compute((5,), lambda x: matrix_B[x][0:8])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros((5,))
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)
    golden = np_A[6][1][2:7] & 0xFF
    golden = golden.astype('uint8')

    ret = hcl_B.asnumpy()
    ret = ret.astype('uint8')
    for i in range(0, 5):
        x = np.unpackbits(golden[i])
        x = np.flip(x)
        y = np.unpackbits(ret[i])
        assert np.array_equal(x, y)

def test_set_bitslice_1D_tensorslice():

    hcl.init()

    def kernel(A, B):
        matrix_C = B[1][2:5][2][4][1:11]
        with hcl.for_(0, 10) as i:
            matrix_C[i][2:0] = A[i]
        return hcl.compute((10,), lambda x: matrix_C[x])

    A = hcl.placeholder((10,))
    B = hcl.placeholder((3, 6, 5, 13))
    C = hcl.placeholder((10,))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(2, size=(10,))
    np_B = np.random.randint(10, size=(3, 6, 5, 13))
    np_C = np.zeros(10)
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_C = hcl.asarray(np_C)

    f(hcl_A, hcl_B, hcl_C)

    golden = (np_B[1][2:5][2][4][1:11] & 0b1100) | np_A
    ret = hcl_C.asnumpy()
    assert np.array_equal(golden, ret)

def test_set_bitslice_2D_tensorslice():

    hcl.init()

    def kernel(A, B):
        matrix_C = B[1][2:5][2][3:7]
        with hcl.for_(0, 5) as x:
            with hcl.for_(0, 10) as y:
                matrix_C[x][y][2:0] = A[x][y]
        return hcl.compute((5,10), lambda x, y: matrix_C[x][y])

    A = hcl.placeholder((5, 10))
    B = hcl.placeholder((3, 6, 8, 10))
    C = hcl.placeholder((5, 10))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(2, size=(5, 10))
    np_B = np.random.randint(10, size=(3, 6, 8, 10))
    np_C = np.zeros((5, 10))
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_C = hcl.asarray(np_C)

    f(hcl_A, hcl_B, hcl_C)

    golden = (np_B[1][2:5][2][3:8] & 0b1100) | np_A
    ret = hcl_C.asnumpy()
    assert np.array_equal(golden, ret)

def test_set_1D():

    hcl.init()

    def kernel(A):
        matrix_B = A[2:8]
        matrix_B[3] = 999
        return hcl.compute((6,), lambda x: matrix_B[x])

    A = hcl.placeholder((10,))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(6)
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    golden = np_A[2:8]
    golden[3] = 999
    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_set_3D():

    hcl.init()

    def kernel(A):
        matrix_B = A[3:8]
        matrix_B[2][7][4] = 999
        return hcl.compute((5, 9, 8), lambda x, y, z: matrix_B[x][y][z])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros((5, 9, 8))
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    golden = np_A[3:8]
    golden[2][7][4] = 999
    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_set_2D():

    hcl.init()

    def kernel(A):
        matrix_B = A[0][3:8]
        matrix_B[2][4] = 999
        return hcl.compute((5, 8), lambda x, y: matrix_B[x][y])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros((5, 8))
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    golden = np_A[0][3:8]
    golden[2][4] = 999
    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_set_1D_tensorslice():

    hcl.init()

    def kernel(A):
        matrix_B = A[0][1][3:8]
        matrix_B[4] = 999
        return hcl.compute((5,), lambda x: matrix_B[x])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros(5)
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    golden = np_A[0][1][3:8]
    golden[4] = 999
    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_multi_slice():

    hcl.init()

    def kernel(A):
        return hcl.compute((3,8), lambda x, y: A[2:7][1][3:6][x][y])

    A = hcl.placeholder((10, 9, 8))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8))
    np_B = np.zeros((3, 8))
    golden = np_A[2:7][1][3:6]
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_multislice_copy_3D():

    hcl.init()

    def kernel(A):
        matrix_C = hcl.copy(A[5][2:7][4][1:3])
        A[5][2:7][4][1:3] = 999
        with hcl.for_(0, 2) as x:
            with hcl.for_(0, 11) as y:
                with hcl.for_(0, 12) as z:
                    hcl.assert_(A[5][6][x+1][y][z] == 999)
        return hcl.compute((2,11,12), lambda x,y,z: matrix_C[x][y][z])

    A = hcl.placeholder((6, 9, 8, 11, 12))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 9, 8, 11, 12))
    np_B = np.zeros((2,11,12))
    golden = np_A[5][2:7][4][1:3]
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_multislice_copy_4D():

    hcl.init()

    def kernel(A):
        matrix_C = hcl.copy(A[5][2:7][1:3])
        A[5][2:7][1:3] = 999
        with hcl.for_(0, 2) as x:
            with hcl.for_(0, 4) as y:
                with hcl.for_(0, 3) as z:
                    with hcl.for_(0, 2) as a:
                        hcl.assert_(A[5][x+3][y][z][a] == 999)
        return hcl.compute((2, 4, 3, 2), lambda x,y,z,a: matrix_C[x][y][z][a])

    A = hcl.placeholder((8, 7, 4, 3, 2))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(8, 7, 4, 3, 2))
    np_B = np.zeros((2, 4, 3, 2))
    golden = np_A[5][2:7][1:3]
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_multislice_copy_5D():

    hcl.init()

    def kernel(A):
        matrix_C = hcl.copy(A[2:7][1:3])
        A[2:7][1:3] = 999
        with hcl.for_(0, 2) as x:
            with hcl.for_(0, 3) as y:
                with hcl.for_(0, 4) as z:
                    with hcl.for_(0, 3) as a:
                        with hcl.for_(0, 2) as b:
                            hcl.assert_(A[x+3][y][z][a][b] == 999)
        return hcl.compute((2, 3, 4, 3, 2), lambda x,y,z,a,b: matrix_C[x][y][z][a][b])

    A = hcl.placeholder((8, 3, 4, 3, 2))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(8, 3, 4, 3, 2))
    np_B = np.zeros((2, 3, 4, 3, 2))
    golden = np_A[2:7][1:3]
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    ret = hcl_B.asnumpy()
    assert np.array_equal(golden, ret)

def test_set_tensorslice_1D_nested():

    hcl.init()

    def kernel(A, B):
        A[1:5] = B[1][3:5][0][2][4:8]
        return hcl.compute((10,), lambda x: A[x])

    A = hcl.placeholder((10,))
    B = hcl.placeholder((3, 6, 5, 13))
    C = hcl.placeholder((10,))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(20, size=(10,))
    np_B = np.random.randint(10, size=(3, 6, 5, 13))
    np_C = np.zeros(10)
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_C = hcl.asarray(np_C)
    f(hcl_A, hcl_B, hcl_C)

    golden = np_A
    golden[1:5] = np_B[1][3:5][0][2][4:8]
    ret = hcl_C.asnumpy()
    assert np.array_equal(golden, ret)

def test_set_tensorslice_3D():

    hcl.init()

    def kernel(A, B):
        A[1][2][2:6] = B[1:5]
        return hcl.compute((2,4,8,3,4), lambda x,y,z,a,b: A[x][y][z][a][b])

    A = hcl.placeholder((2, 4, 8, 3, 4))
    B = hcl.placeholder((6,3, 4))
    C = hcl.placeholder((2,4,8,3, 4))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(20, size=(2, 4, 8, 3, 4))
    np_B = np.random.randint(10, size=(6,3, 4))
    np_C = np.zeros((2, 4, 8, 3, 4))
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_C = hcl.asarray(np_C)

    f(hcl_A, hcl_B, hcl_C)

    golden = np_A
    golden[1][2][2:6] = np_B[1:5]
    ret = hcl_C.asnumpy()
    assert np.array_equal(golden, ret)

def test_set_tensorslice_3D_nested():

    hcl.init()

    def kernel(A, B):
        A[1][2][2:6][3][1:2] = B[2:5][1][1:2]
        return hcl.compute((2,4,8,3,4), lambda x,y,z,a,b: A[x][y][z][a][b])

    A = hcl.placeholder((2, 4, 8, 3, 4))
    B = hcl.placeholder((6,3, 4))
    C = hcl.placeholder((2,4,8,3, 4))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(20, size=(2, 4, 8, 3, 4))
    np_B = np.random.randint(10, size=(6,3, 4))
    np_C = np.zeros((2, 4, 8, 3, 4))
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_C = hcl.asarray(np_C)

    f(hcl_A, hcl_B, hcl_C)

    golden = np_A
    golden[1][2][2:6][3][1:2] = np_B[2:5][1][1:2]
    ret = hcl_C.asnumpy()
    assert np.array_equal(golden, ret)

def test_set_tensorslice_2D_tensor():

    hcl.init()

    def kernel(A, B):
        A[1][2][2:6][3][1:2] = B
        return hcl.compute((2,4,8,3,4), lambda x,y,z,a,b: A[x][y][z][a][b])

    A = hcl.placeholder((2, 4, 8, 3, 4))
    B = hcl.placeholder((1, 4))
    C = hcl.placeholder((2,4,8,3, 4))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(20, size=(2, 4, 8, 3, 4))
    np_B = np.random.randint(10, size=(1, 4))
    np_C = np.zeros((2, 4, 8, 3, 4))
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_C = hcl.asarray(np_C)

    f(hcl_A, hcl_B, hcl_C)

    golden = np_A
    golden[1][2][2:6][3][1:2] = np_B
    ret = hcl_C.asnumpy()
    assert np.array_equal(golden, ret)
