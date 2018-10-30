import heterocl as hcl
import numpy as np

def test_pipeline():
    hcl.init()
    initiation_interval = 4
    a = hcl.placeholder((10, 20))
    b = hcl.placeholder((10, 20))
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j])
    s = hcl.create_schedule(c)
    s[c].pipeline(c.axis[0], initiation_interval)
    ir = hcl.lower(s, [a, b, c])
    pipeline_hint_str = "\"initiation_interval\"="+str(initiation_interval)
    assert pipeline_hint_str in str(ir)


def test_unroll():
    hcl.init()
    factor = 4
    a = hcl.placeholder((10, 20))
    b = hcl.placeholder((10, 20))
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j])
    s = hcl.create_schedule(c)
    s[c].unroll(c.axis[0], factor=factor)
    ir = hcl.lower(s, [a, b, c])
    unroll_hint_str = "\"factor\"="+str(factor)
    assert unroll_hint_str in str(ir)


def test_fuse():
    hcl.init()
    a = hcl.placeholder((10, 20, 30, 40))
    b = hcl.placeholder((10, 20, 30, 40))
    c = hcl.compute(a.shape, lambda i, j, k, l: a[i, j, k, l] + b[i, j, k, l])
    s = hcl.create_schedule(c)
    s[c].fuse(c.axis[1], c.axis[2])
    ir = hcl.lower(s, [a, b, c])
    assert "j.k.fused" in str(ir)


def test_reorder():
    hcl.init()
    a = hcl.placeholder((10, 20, 30, 40), name="a")
    b = hcl.placeholder((10, 20, 30, 40), name="b")
    c = hcl.compute(a.shape, lambda i, j, k, l: a[i, j, k, l] + b[i, j, k, l], name="c")

    # axes are consecutive
    def test_case_1():
        s = hcl.create_schedule(c)
        s[c].reorder(c.axis[2], c.axis[1])
        ir = hcl.lower(s, [a, b, c])
        assert str(ir.body.body).startswith("for (i, 0, 10)")
        assert str(ir.body.body.body).startswith("for (k, 0, 30)")
        assert str(ir.body.body.body.body).startswith("for (j, 0, 20)")
        assert str(ir.body.body.body.body.body).startswith("for (l, 0, 40)")

    # axes are not consecutive
    def test_case_2():
        s = hcl.create_schedule(c)
        s[c].reorder(c.axis[3], c.axis[0])
        ir = hcl.lower(s, [a, b, c])
        assert str(ir.body.body).startswith("for (l, 0, 40)")
        assert str(ir.body.body.body).startswith("for (j, 0, 20)")
        assert str(ir.body.body.body.body).startswith("for (k, 0, 30)")
        assert str(ir.body.body.body.body.body).startswith("for (i, 0, 10)")

    test_case_1()
    test_case_2()


def test_split():
    hcl.init()
    a = hcl.placeholder((10, 20), name="a")
    b = hcl.placeholder((10, 20), name="b")
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")

    def test_transform_mode():
        s = hcl.create_schedule(c)
        s[c].split(c.axis[1], factor=3, mode="transform")
        ir = hcl.lower(s, [a, b, c])
        assert str(ir.body.body).startswith("for (i, 0, 10)")
        assert str(ir.body.body.body).startswith("for (j.outer, 0, 7)")
        assert str(ir.body.body.body.body).startswith("for (j.inner, 0, 3)")
        assert str(ir.body.body.body.body.body).startswith(
            "if (((j.outer*3) < (20 - j.inner)))")

    def test_annotate_mode():
        split_factor = 3
        s = hcl.create_schedule(c)
        s[c].split(c.axis[1], factor=split_factor, mode="annotate")
        split_hint_str = "\"split_factor\"="+str(split_factor)
        ir = hcl.lower(s, [a, b, c])
        assert split_hint_str in str(ir)

    test_transform_mode()
    test_annotate_mode()


def test_split_reorder():
    hcl.init()
    a = hcl.placeholder((10, 20), name="a")
    b = hcl.placeholder((10, 20), name="b")
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")

    def test_case_1():
        s = hcl.create_schedule(c)
        yo, yi = s[c].split(c.axis[0], factor=3, mode="transform")
        xo, xi = s[c].split(c.axis[1], factor=3, mode="transform")
        s[c].reorder(yo, xo, yi, xi)
        ir = hcl.lower(s, [a, b, c])
        assert str(ir.body.body).startswith("for (i.outer, 0, 4)")
        assert str(ir.body.body.body).startswith("for (j.outer, 0, 7)")
        assert str(ir.body.body.body.body).startswith("for (i.inner, 0, 3)")
        assert str(ir.body.body.body.body.body).startswith(
            "if (((i.outer*3) < (10 - i.inner)))")
        assert str(ir.body.body.body.body.body.then_case).startswith(
            "for (j.inner, 0, 3)")
        assert str(ir.body.body.body.body.body.then_case.body).startswith(
            "if (((j.outer*3) < (20 - j.inner)))")

    def test_case_2():
        s = hcl.create_schedule(c)
        yo, yi = s[c].split(c.axis[0], factor=3, mode="transform")
        xo, xi = s[c].split(c.axis[1], factor=3, mode="transform")
        s[c].reorder(yi, xi, yo, xo)
        ir = hcl.lower(s, [a, b, c])
        assert str(ir.body.body).startswith("for (i.inner, 0, 3)")
        assert str(ir.body.body.body).startswith("for (j.inner, 0, 3)")
        assert str(ir.body.body.body.body).startswith("for (i.outer, 0, 4)")
        assert str(ir.body.body.body.body.body).startswith(
            "if (((i.outer*3) < (10 - i.inner)))")
        assert str(ir.body.body.body.body.body.then_case).startswith(
            "for (j.outer, 0, 7)")
        assert str(ir.body.body.body.body.body.then_case.body).startswith(
            "if (((j.outer*3) < (20 - j.inner)))")

    test_case_1()
    test_case_2()


def test_compute_at():
    hcl.init()
    A = hcl.placeholder((10, 20, 30), name="A")
    B = hcl.compute(A.shape, lambda i, j, m: A[i, j, m] * 2, name="B")
    C = hcl.compute(B.shape, lambda ii, jj, mm: B[ii, jj, mm] + 1, name="C")

    def _verify_build(sch):
        f = hcl.build(sch, [A, C])
        a_np = np.random.randint(low=0, high=100, size=A.shape)
        a_hcl = hcl.asarray(a_np)
        c_hcl = hcl.asarray(np.zeros(C.shape), dtype="int32")
        f(a_hcl, c_hcl)
        c_np = a_np * 2 + 1
        np.testing.assert_allclose(c_np, c_hcl.asnumpy())

    def test_case_1():
        # axis 0
        s0 = hcl.create_schedule(C)
        s0[B].compute_at(s0[C], C.axis[0])
        ir0 = hcl.lower(s0, [A, C])
        assert "allocate B[int32 * 1 * 20 * 30]" in str(ir0)
        _verify_build(s0)
        # axis 1
        s1 = hcl.create_schedule(C)
        s1[B].compute_at(s1[C], C.axis[1])
        ir1 = hcl.lower(s1, [A, C])
        assert "allocate B[int32 * 1 * 1 * 30]" in str(ir1)
        _verify_build(s1)
        # axis 2
        s2 = hcl.create_schedule(C)
        s2[B].compute_at(s2[C], C.axis[2])
        ir2 = hcl.lower(s2, [A, C])
        assert "allocate B[int32 * 1 * 1 * 1]" in str(ir2)
        _verify_build(s2)

    def test_case_2():
        s = hcl.create_schedule(C)
        s[B].compute_at(s[C], C.axis[2])
        s[C].fuse(C.axis[0], C.axis[1])
        ir = hcl.lower(s, [A, C])
        assert "allocate B[int32 * 1 * 1 * 1]" in str(ir)
        _verify_build(s)

    def test_case_3():
        s = hcl.create_schedule(C)
        s[B].compute_at(s[C], C.axis[2])
        s[C].split(C.axis[0], factor=3)
        s[C].split(C.axis[1], factor=3)
        ir = hcl.lower(s, [A, C])
        assert "allocate B[int32 * 1 * 1 * 1]" in str(ir)
        _verify_build(s)

    # compute_at and reorder, compute at an axis that is not reordered
    def test_case_4():
        s0 = hcl.create_schedule(C)
        s0[B].compute_at(s0[C], C.axis[2])
        s0[C].reorder(C.axis[1], C.axis[0])
        ir0 = hcl.lower(s0, [A, C])
        assert "allocate B[int32 * 1 * 1 * 1]" in str(ir0)
        _verify_build(s0)
        s1 = hcl.create_schedule(C)
        s1[B].compute_at(s1[C], C.axis[1])
        s1[C].reorder(C.axis[2], C.axis[0])
        ir1 = hcl.lower(s1, [A, C])
        assert "allocate B[int32 * 1 * 1 * 10]" in str(ir1)
        _verify_build(s1)

    # compute_at and reorder, compute at an axis that has been reordered
    def test_case_5():
        s0 = hcl.create_schedule(C)
        s0[B].compute_at(s0[C], C.axis[1])
        s0[C].reorder(C.axis[1], C.axis[0])
        ir0 = hcl.lower(s0, [A, C])
        assert "allocate B[int32 * 1 * 10 * 30]" in str(ir0)
        _verify_build(s0)
        s1 = hcl.create_schedule(C)
        s1[B].compute_at(s1[C], C.axis[0])
        s1[C].reorder(C.axis[1], C.axis[0])
        ir1 = hcl.lower(s1, [A, C])
        assert "allocate B[int32 * 1 * 1 * 30]" in str(ir1)
        _verify_build(s1)

    def test_case_6():
        s = hcl.create_schedule(C)
        s[B].compute_at(s[C], C.axis[2])
        yo, yi = s[C].split(C.axis[0], factor=3)
        xo, xi = s[C].split(C.axis[1], factor=3)
        s[C].reorder(yo, xo, yi, xi)
        ir = hcl.lower(s, [A, C])
        assert "allocate B[int32 * 1 * 1 * 1]" in str(ir)
        _verify_build(s)

    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()
    test_case_6()


def test_compute_at_complex():
    hcl.init()
    A = hcl.placeholder((10, 20, 30), name="A")
    B = hcl.compute(A.shape, lambda i, j, m: A[i, j, m] * 2, name="B")
    C = hcl.compute(B.shape, lambda ii, jj, mm: B[ii, jj, mm] + 1, name="C")
    D = hcl.compute(C.shape, lambda iii, jjj, mmm: C[iii, jjj, mmm] % 3, name="D")
    s = hcl.create_schedule(D)
    s[B].compute_at(s[D], D.axis[1])
    s[C].compute_at(s[D], D.axis[2])
    ir = hcl.lower(s, [A, D])
    assert "allocate B[int32 * 1 * 1 * 30]" in str(ir)
    assert "allocate C[int32 * 1 * 1 * 1]" in str(ir)
    f = hcl.build(s, [A, D])
    a_np = np.random.randint(low=0, high=100, size=A.shape)
    a_hcl = hcl.asarray(a_np)
    d_hcl = hcl.asarray(np.zeros(D.shape), dtype="int32")
    f(a_hcl, d_hcl)
    d_np = (a_np * 2 + 1) % 3
    np.testing.assert_allclose(d_np, d_hcl.asnumpy())

if __name__ == '__main__':
    test_pipeline()
    test_unroll()
    test_fuse()
    test_reorder()
    test_split()
    test_split_reorder()
    test_compute_at()
    test_compute_at_complex()
