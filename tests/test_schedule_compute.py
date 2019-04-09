import heterocl as hcl
import numpy as np

def test_pipeline():
    hcl.init()
    initiation_interval = 4
    a = hcl.placeholder((10, 20))
    b = hcl.placeholder((10, 20))
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j])
    s = hcl.create_schedule([a, b, c])
    s[c].pipeline(c.axis[0], initiation_interval)
    ir = hcl.lower(s)
    pipeline_hint_str = "\"initiation_interval\"="+str(initiation_interval)
    assert pipeline_hint_str in str(ir)

def test_pipeline_num_axis():
    hcl.init()
    initiation_interval = 4
    a = hcl.placeholder((10, 20))
    b = hcl.placeholder((10, 20))
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j])
    s = hcl.create_schedule([a, b, c])
    s[c].pipeline(0, initiation_interval)
    ir = hcl.lower(s)
    pipeline_hint_str = "\"initiation_interval\"="+str(initiation_interval)
    assert pipeline_hint_str in str(ir)

def test_unroll():
    hcl.init()
    factor = 4
    a = hcl.placeholder((10, 20))
    b = hcl.placeholder((10, 20))
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j])
    s = hcl.create_schedule([a, b, c])
    s[c].unroll(c.axis[0], factor=factor)
    ir = hcl.lower(s)
    unroll_hint_str = "\"factor\"="+str(factor)
    assert unroll_hint_str in str(ir)

def test_unroll_num_axis():
    hcl.init()
    factor = 4
    a = hcl.placeholder((10, 20))
    b = hcl.placeholder((10, 20))
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j])
    s = hcl.create_schedule([a, b, c])
    s[c].unroll(0, factor=factor)
    ir = hcl.lower(s)
    unroll_hint_str = "\"factor\"="+str(factor)
    assert unroll_hint_str in str(ir)

def test_fuse():
    hcl.init()
    a = hcl.placeholder((10, 20, 30, 40))
    b = hcl.placeholder((10, 20, 30, 40))
    c = hcl.compute(a.shape, lambda i, j, k, l: a[i, j, k, l] + b[i, j, k, l])
    s = hcl.create_schedule([a, b, c])
    s[c].fuse(c.axis[1], c.axis[2])
    ir = hcl.lower(s)
    assert "j.k.fused" in str(ir)

def test_fuse_num_axis():
    hcl.init()
    a = hcl.placeholder((10, 20, 30, 40))
    b = hcl.placeholder((10, 20, 30, 40))
    c = hcl.compute(a.shape, lambda i, j, k, l: a[i, j, k, l] + b[i, j, k, l])
    s = hcl.create_schedule([a, b, c])
    s[c].fuse(1, 2)
    ir = hcl.lower(s)
    assert "j.k.fused" in str(ir)

def test_reorder():
    hcl.init()
    a = hcl.placeholder((10, 20, 30, 40), name="a")
    b = hcl.placeholder((10, 20, 30, 40), name="b")
    c = hcl.compute(a.shape, lambda i, j, k, l: a[i, j, k, l] + b[i, j, k, l], name="c")

    # axes are consecutive
    def test_case_1():
        s = hcl.create_schedule([a, b, c])
        s[c].reorder(c.axis[2], c.axis[1])
        ir = hcl.lower(s)
        assert str(ir.body.body).startswith("for (i, 0, 10)")
        assert str(ir.body.body.body).startswith("for (k, 0, 30)")
        assert str(ir.body.body.body.body).startswith("for (j, 0, 20)")
        assert str(ir.body.body.body.body.body).startswith("for (l, 0, 40)")

    # axes are not consecutive
    def test_case_2():
        s = hcl.create_schedule([a, b, c])
        s[c].reorder(c.axis[3], c.axis[0])
        ir = hcl.lower(s)
        assert str(ir.body.body).startswith("for (l, 0, 40)")
        assert str(ir.body.body.body).startswith("for (j, 0, 20)")
        assert str(ir.body.body.body.body).startswith("for (k, 0, 30)")
        assert str(ir.body.body.body.body.body).startswith("for (i, 0, 10)")

    test_case_1()
    test_case_2()

def test_reorder_num_axis():
    hcl.init()
    a = hcl.placeholder((10, 20, 30, 40), name="a")
    b = hcl.placeholder((10, 20, 30, 40), name="b")
    c = hcl.compute(a.shape, lambda i, j, k, l: a[i, j, k, l] + b[i, j, k, l], name="c")

    s = hcl.create_schedule([a, b, c])
    s[c].reorder(2, 1)
    ir = hcl.lower(s)
    assert str(ir.body.body).startswith("for (i, 0, 10)")
    assert str(ir.body.body.body).startswith("for (k, 0, 30)")
    assert str(ir.body.body.body.body).startswith("for (j, 0, 20)")
    assert str(ir.body.body.body.body.body).startswith("for (l, 0, 40)")

def test_split():
    hcl.init()
    a = hcl.placeholder((10, 20), name="a")
    b = hcl.placeholder((10, 20), name="b")
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")

    # without if condition
    def test_transform_mode_1():
        s = hcl.create_schedule([a, b, c])
        s[c].split(c.axis[1], factor=4, mode="transform")
        ir = hcl.lower(s)
        assert str(ir.body.body).startswith("for (i, 0, 10)")
        assert str(ir.body.body.body).startswith("for (j.outer, 0, 5)")
        assert str(ir.body.body.body.body).startswith("for (j.inner, 0, 4)")
        assert str(ir.body.body.body.body.body).startswith("c[")

    # with if condition
    def test_transform_mode_2():
        s = hcl.create_schedule([a, b, c])
        s[c].split(c.axis[1], factor=3, mode="transform")
        ir = hcl.lower(s)
        assert str(ir.body.body).startswith("for (i, 0, 10)")
        assert str(ir.body.body.body).startswith("for (j.outer, 0, 7)")
        assert str(ir.body.body.body.body).startswith("for (j.inner, 0, 3)")
        assert str(ir.body.body.body.body.body).startswith(
            "if (((j.outer*3) < (20 - j.inner)))")

    def test_annotate_mode():
        split_factor = 3
        s = hcl.create_schedule([a, b, c])
        s[c].split(c.axis[1], factor=split_factor, mode="annotate")
        split_hint_str = "\"split_factor\"="+str(split_factor)
        ir = hcl.lower(s)
        assert split_hint_str in str(ir)

    test_transform_mode_1()
    test_transform_mode_2()
    test_annotate_mode()

def test_split_num_axis():
    hcl.init()
    a = hcl.placeholder((10, 20), name="a")
    b = hcl.placeholder((10, 20), name="b")
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")

    s = hcl.create_schedule([a, b, c])
    s[c].split(1, factor=4, mode="transform")
    ir = hcl.lower(s)
    assert str(ir.body.body).startswith("for (i, 0, 10)")
    assert str(ir.body.body.body).startswith("for (j.outer, 0, 5)")
    assert str(ir.body.body.body.body).startswith("for (j.inner, 0, 4)")
    assert str(ir.body.body.body.body.body).startswith("c[")

def test_split_reorder():
    hcl.init()
    a = hcl.placeholder((10, 20), name="a")
    b = hcl.placeholder((10, 20), name="b")
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")

    def test_case_1():
        s = hcl.create_schedule([a, b, c])
        xo, xi = s[c].split(c.axis[0], factor=2, mode="transform")
        yo, yi = s[c].split(c.axis[1], factor=5, mode="transform")
        s[c].reorder(yo, xo, yi, xi)
        ir = hcl.lower(s)
        assert str(ir.body.body).startswith("for (j.outer, 0, 4)")
        assert str(ir.body.body.body).startswith("for (i.outer, 0, 5)")
        assert str(ir.body.body.body.body).startswith("for (j.inner, 0, 5)")
        assert str(ir.body.body.body.body.body).startswith("for (i.inner, 0, 2)")

    def test_case_2():
        s = hcl.create_schedule([a, b, c])
        xo, xi = s[c].split(c.axis[0], factor=3, mode="transform")
        yo, yi = s[c].split(c.axis[1], factor=3, mode="transform")
        s[c].reorder(yi, xi, yo, xo)
        ir = hcl.lower(s)
        assert str(ir.body.body).startswith("for (j.inner, 0, 3)")
        assert str(ir.body.body.body).startswith("for (i.inner, 0, 3)")
        assert str(ir.body.body.body.body).startswith("for (j.outer, 0, 7)")
        assert str(ir.body.body.body.body.body).startswith("for (i.outer, 0, 4)")
        assert str(ir.body.body.body.body.body.body).startswith(
            "if (((j.outer*3) < (20 - j.inner)))")
        assert str(ir.body.body.body.body.body.body.then_case).startswith(
            "if (((i.outer*3) < (10 - i.inner)))")

    test_case_1()
    test_case_2()

def test_split_reorder_num_axis():
    # note that this is not the recommanded way
    hcl.init()
    a = hcl.placeholder((10, 20), name="a")
    b = hcl.placeholder((10, 20), name="b")
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")

    s = hcl.create_schedule([a, b, c])
    xo, xi = s[c].split(0, factor=2, mode="transform")
    yo, yi = s[c].split(2, factor=5, mode="transform")
    s[c].reorder(2, 0, 3, 1)
    ir = hcl.lower(s)
    assert str(ir.body.body).startswith("for (j.outer, 0, 4)")
    assert str(ir.body.body.body).startswith("for (i.outer, 0, 5)")
    assert str(ir.body.body.body.body).startswith("for (j.inner, 0, 5)")
    assert str(ir.body.body.body.body.body).startswith("for (i.inner, 0, 2)")

def test_compute_at():
    hcl.init()
    A = hcl.placeholder((10, 20, 30), name="A")
    B = hcl.compute(A.shape, lambda i, j, m: A[i, j, m] * 2, name="B")
    C = hcl.compute(B.shape, lambda ii, jj, mm: B[ii, jj, mm] + 1, name="C")

    def _verify_build(sch):
        f = hcl.build(sch)
        a_np = np.random.randint(low=0, high=100, size=A.shape)
        a_hcl = hcl.asarray(a_np)
        c_hcl = hcl.asarray(np.zeros(C.shape), dtype="int32")
        f(a_hcl, c_hcl)
        c_np = a_np * 2 + 1
        np.testing.assert_allclose(c_np, c_hcl.asnumpy())

    def test_case_1():
        # axis 0
        s0 = hcl.create_schedule([A, C])
        s0[B].compute_at(s0[C], C.axis[0])
        ir0 = hcl.lower(s0)
        assert "allocate B[int32 * 1 * 20 * 30]" in str(ir0)
        _verify_build(s0)
        # axis 1
        s1 = hcl.create_schedule([A, C])
        s1[B].compute_at(s1[C], C.axis[1])
        ir1 = hcl.lower(s1)
        assert "allocate B[int32 * 1 * 1 * 30]" in str(ir1)
        _verify_build(s1)
        # axis 2
        s2 = hcl.create_schedule([A, C])
        s2[B].compute_at(s2[C], C.axis[2])
        ir2 = hcl.lower(s2)
        assert "allocate B[int32 * 1 * 1 * 1]" in str(ir2)
        _verify_build(s2)

    def test_case_2():
        s = hcl.create_schedule([A, C])
        s[B].compute_at(s[C], C.axis[2])
        s[C].fuse(C.axis[0], C.axis[1])
        ir = hcl.lower(s)
        assert "allocate B[int32 * 1 * 1 * 1]" in str(ir)
        _verify_build(s)

    def test_case_3():
        s = hcl.create_schedule([A, C])
        s[B].compute_at(s[C], C.axis[2])
        s[C].split(C.axis[0], factor=3)
        s[C].split(C.axis[1], factor=3)
        ir = hcl.lower(s)
        assert "allocate B[int32 * 1 * 1 * 1]" in str(ir)
        _verify_build(s)

    # compute_at and reorder, compute at an axis that is not reordered
    # check both directions of reorder and compute_at
    def test_case_4():
        s0 = hcl.create_schedule([A, C])
        s0[B].compute_at(s0[C], C.axis[2])
        s0[C].reorder(C.axis[1], C.axis[0])
        ir0 = hcl.lower(s0)
        assert "allocate B[int32 * 1 * 1 * 1]" in str(ir0)
        _verify_build(s0)

    # compute_at and reorder, compute at an axis that has been reordered
    # note that the results will be different
    def test_case_5():
        s0 = hcl.create_schedule([A, C])
        s0[B].compute_at(s0[C], C.axis[1])
        s0[C].reorder(C.axis[1], C.axis[0])
        ir0 = hcl.lower(s0)
        assert "allocate B[int32 * 1 * 1 * 30]" in str(ir0)
        _verify_build(s0)

    def test_case_6():
        s = hcl.create_schedule([A, C])
        s[B].compute_at(s[C], C.axis[2])
        yo, yi = s[C].split(C.axis[0], factor=3)
        xo, xi = s[C].split(C.axis[1], factor=3)
        s[C].reorder(yo, xo, yi, xi)
        ir = hcl.lower(s)
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
    s = hcl.create_schedule([A, D])
    s[B].compute_at(s[C], C.axis[1])
    s[C].compute_at(s[D], D.axis[2])
    ir = hcl.lower(s)
    assert "allocate B[int32 * 1 * 1 * 30]" in str(ir)
    assert "allocate C[int32 * 1 * 1 * 1]" in str(ir)
    f = hcl.build(s)
    a_np = np.random.randint(low=0, high=100, size=A.shape)
    a_hcl = hcl.asarray(a_np)
    d_hcl = hcl.asarray(np.zeros(D.shape), dtype="int32")
    f(a_hcl, d_hcl)
    d_np = (a_np * 2 + 1) % 3
    np.testing.assert_allclose(d_np, d_hcl.asnumpy())

def test_compute_at_complex_num_axis():
    hcl.init()
    A = hcl.placeholder((10, 20, 30), name="A")
    B = hcl.compute(A.shape, lambda i, j, m: A[i, j, m] * 2, name="B")
    C = hcl.compute(B.shape, lambda ii, jj, mm: B[ii, jj, mm] + 1, name="C")
    D = hcl.compute(C.shape, lambda iii, jjj, mmm: C[iii, jjj, mmm] % 3, name="D")
    s = hcl.create_schedule([A, D])
    s[B].compute_at(s[C], 1)
    s[C].compute_at(s[D], 2)
    ir = hcl.lower(s)
    assert "allocate B[int32 * 1 * 1 * 30]" in str(ir)
    assert "allocate C[int32 * 1 * 1 * 1]" in str(ir)
    f = hcl.build(s)
    a_np = np.random.randint(low=0, high=100, size=A.shape)
    a_hcl = hcl.asarray(a_np)
    d_hcl = hcl.asarray(np.zeros(D.shape), dtype="int32")
    f(a_hcl, d_hcl)
    d_np = (a_np * 2 + 1) % 3
    np.testing.assert_allclose(d_np, d_hcl.asnumpy())

def test_multi_stage():
    hcl.init()
    def test(A):
        r = hcl.reduce_axis(0, 10)
        B = hcl.compute((10,), lambda x: hcl.sum(A[x, r], axis=r), "B")
        return B
    A = hcl.placeholder((10, 10))
    s = hcl.create_schedule([A], test)
    s[test.B].split(test.B.axis[0], 5)
    f = hcl.build(s)
    a_np = np.random.randint(0, 10, size=(10, 10))
    b_np = np.zeros(shape=(10,), dtype="int")
    a_hcl = hcl.asarray(a_np)
    b_hcl = hcl.asarray(b_np)
    f(a_hcl, b_hcl)
    d_np = np.sum(a_np, axis=1)
    np.testing.assert_array_equal(d_np, b_hcl.asnumpy())
