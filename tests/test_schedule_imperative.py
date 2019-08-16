import heterocl as hcl
import numpy as np


def test_if():

    hcl.init()
    def absolute(A, B):
        with hcl.for_(0, A.shape[0], name="x") as x:
            with hcl.for_(0, A.shape[1], name="y") as y:
                with hcl.if_(A[x, y] >= 0):
                    B[x, y] = A[x, y]
                with hcl.else_():
                    B[x, y] = -A[x, y]

    A = hcl.placeholder((10, 20), name="A", dtype="float32")
    B = hcl.placeholder(A.shape, name="B", dtype="float32")
    with hcl.Stage() as C:
        absolute(A, B)
    s = hcl.create_schedule([A, B])
    o, i = s[C].split(C.x, factor=3)
    s[C].reorder(i, o)
    # test lower
    ir = hcl.lower(s)
    assert str(ir.body.body.body.body).startswith("for (x.inner, 0, 3)")
    assert str(ir.body.body.body.body.body).startswith("for (x.outer, 0, 4)")
    assert str(ir.body.body.body.body.body.body).startswith(
        "for (y, 0, 20)")
    assert str(ir.body.body.body.body.body.body.body.condition).startswith(
        "(x.inner < (10 - (x.outer*3)))")
    assert str(ir.body.body.body.body.body.body.body.then_case.condition).startswith(
        "(0.000000f <= A[(y + ((x.inner + (x.outer*3))*20))])")
    assert str(ir.body.body.body.body.body.body.body.then_case.then_case).startswith(
        "B[(y + ((x.inner + (x.outer*3))*20))] = A[(y + ((x.inner + (x.outer*3))*20))]")
    assert str(ir.body.body.body.body.body.body.body.then_case.else_case).startswith(
        "B[(y + ((x.inner + (x.outer*3))*20))] = (A[(y + ((x.inner + (x.outer*3))*20))]*-1.000000f)")
    # test build
    f = hcl.build(s)
    a_np = np.random.random((A.shape))
    a_hcl = hcl.asarray(a_np, dtype="float32")
    b_hcl = hcl.asarray(np.zeros(B.shape), dtype="float32")
    f(a_hcl, b_hcl)
    b_np = np.abs(a_np)
    np.testing.assert_allclose(b_np, b_hcl.asnumpy())


def test_schedule_intra_stage():

    hcl.init()
    def popcount(A, B): # each element in A is a 32-bit integer
        with hcl.for_(0, A.shape[0], name="x") as x:
            with hcl.for_(0, A.shape[1], name="y") as y:
                B[x, y] = 0
                with hcl.for_(0, 32) as i:
                    B[x, y] += A[x, y][i]

    A = hcl.placeholder((10, 20))
    B = hcl.placeholder(A.shape)
    with hcl.Stage() as C:
        popcount(A, B)

    def test_unroll():
        s = hcl.create_schedule([A, B])
        s[C].unroll(C.x, factor=3)
        ir = hcl.lower(s)
        assert "unrolled \"factor\"=3" in str(ir)

    def test_reorder():
        s = hcl.create_schedule([A, B])
        s[C].reorder(C.y, C.x)
        ir = hcl.lower(s)
        assert str(ir.body.body.body.body).startswith("for (y, 0, 20)")
        assert str(ir.body.body.body.body.body).startswith("for (x, 0, 10)")

    def test_fuse():
        s = hcl.create_schedule([A, B])
        s[C].fuse(C.x, C.y)
        ir = hcl.lower(s)
        assert str(ir.body.body.body.body).startswith("for (x.y.fused, 0, 200)")

    def test_split():
        s = hcl.create_schedule([A, B])
        s[C].split(C.x, factor=3)
        ir = hcl.lower(s)
        assert str(ir.body.body.body.body).startswith("for (x.outer, 0, 4)")
        assert str(ir.body.body.body.body.body).startswith("for (x.inner, 0, 3)")
        assert str(ir.body.body.body.body.body.body).startswith(
            "for (y, 0, 20)")
        assert str(ir.body.body.body.body.body.body.body).startswith(
            "if ((x.inner < (10 - (x.outer*3))))")

    test_unroll()
    test_reorder()
    test_fuse()
    test_split()


def test_schedule_inter_stage():

    hcl.init()
    def popcount(A, B): # each element in A is a 32-bit integer
        with hcl.for_(0, A.shape[0], name="x") as x:
            with hcl.for_(0, A.shape[1], name="y") as y:
                B[x, y] = 0
                with hcl.for_(0, 32) as i:
                    B[x, y] += A[x, y][i]

    A = hcl.placeholder((10, 20))
    B = hcl.compute(A.shape, lambda xx, yy: A[xx, yy] + 1, name="B")
    C = hcl.placeholder(B.shape)
    with hcl.Stage() as Out:
        popcount(B, C)

    def test_compute_at():
        s = hcl.create_schedule([A, C])
        s[B].compute_at(s[Out], Out.y)
        ir = hcl.lower(s)
        assert "allocate B[int32 * 1 * 1]" in str(ir)

    test_compute_at()


if __name__ == '__main__':
    test_if()
    test_schedule_intra_stage()
    test_schedule_inter_stage()
