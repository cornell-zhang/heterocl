import heterocl as hcl
import numpy as np

def test_reuse_blur_x_tensor():
    hcl.init()
    A = hcl.placeholder((10, 10))

    def kernel(A):
        X = hcl.compute((10, 10), lambda y, x: A[y, x], "X")
        B = hcl.compute((10, 8), lambda y, x: X[y, x] + X[y, x+1] + X[y, x+2], "B")
        return B

    s = hcl.create_schedule([A], kernel)
    RB = s.reuse_at(kernel.X, s[kernel.B], kernel.B.axis[1])
    f = hcl.build(s)

    np_A = np.random.randint(0, 10, size=(10, 10))
    np_B = np.zeros((10, 8), dtype="int")
    np_C = np.zeros((10, 8), dtype="int")

    for y in range(0, 10):
        for x in range(0, 8):
            np_C[y][x] = np_A[y][x] + np_A[y][x+1] + np_A[y][x+2]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.array_equal(np_B, np_C)

