# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np
import hcl_mlir
import pytest


def test_pipeline():
    hcl.init()
    initiation_interval = 4
    a = hcl.placeholder((10, 20))
    b = hcl.placeholder((10, 20))
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j])
    s = hcl.create_schedule([a, b])
    s[c].pipeline(c.axis[0], initiation_interval)
    ir = hcl.lower(s)
    pipeline_hint_str = "pipeline_ii = " + str(initiation_interval)
    assert pipeline_hint_str in str(ir)


def test_pipeline_num_axis():
    hcl.init()
    initiation_interval = 4
    a = hcl.placeholder((10, 20))
    b = hcl.placeholder((10, 20))
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j])
    s = hcl.create_schedule([a, b])
    s[c].pipeline(0, initiation_interval)
    ir = hcl.lower(s)
    pipeline_hint_str = "pipeline_ii = " + str(initiation_interval)
    assert pipeline_hint_str in str(ir)


def test_unroll():
    hcl.init()
    factor = 4
    a = hcl.placeholder((10, 20))
    b = hcl.placeholder((10, 20))
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j])
    s = hcl.create_schedule([a, b])
    s[c].unroll(c.axis[0], factor=factor)
    ir = hcl.lower(s)
    unroll_hint_str = "unroll = " + str(factor)
    assert unroll_hint_str in str(ir)


def test_unroll_num_axis():
    hcl.init()
    factor = 4
    a = hcl.placeholder((10, 20))
    b = hcl.placeholder((10, 20))
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j])
    s = hcl.create_schedule([a, b])
    s[c].unroll(0, factor=factor)
    ir = hcl.lower(s)
    unroll_hint_str = "unroll = " + str(factor)
    assert unroll_hint_str in str(ir)


def test_fuse():
    hcl.init()
    a = hcl.placeholder((10, 20, 30, 40))
    b = hcl.placeholder((10, 20, 30, 40))
    c = hcl.compute(a.shape, lambda i, j, k, l: a[i, j, k, l] + b[i, j, k, l])
    s = hcl.create_schedule([a, b])
    s[c].fuse(c.axis[1], c.axis[2])
    ir = hcl.lower(s)
    assert "j_k_fused" in str(ir)


def test_fuse_num_axis():
    hcl.init()
    a = hcl.placeholder((10, 20, 30, 40))
    b = hcl.placeholder((10, 20, 30, 40))
    c = hcl.compute(a.shape, lambda i, j, k, l: a[i, j, k, l] + b[i, j, k, l])
    s = hcl.create_schedule([a, b])
    s[c].fuse(1, 2)
    ir = hcl.lower(s)
    assert "j_k_fused" in str(ir)


def test_reorder():
    def algo():
        hcl.init()
        a = hcl.placeholder((10, 20, 30, 40), name="a")
        b = hcl.placeholder((10, 20, 30, 40), name="b")
        c = hcl.compute(
            a.shape, lambda i, j, k, l: a[i, j, k, l] + b[i, j, k, l], name="c"
        )
        return a, b, c

    # axes are consecutive
    def test_case_1():
        a, b, c = algo()
        s = hcl.create_schedule([a, b])
        s[c].reorder(c.axis[2], c.axis[1])
        ir = hcl.lower(s)
        loops = hcl_mlir.get_affine_loop_nests(s.top_func)[0]
        assert "i" in str(loops[0]["name"])
        assert "0 to 10" in str(loops[0]["body"])
        assert "k" in str(loops[1]["name"])
        assert "0 to 30" in str(loops[1]["body"])
        assert "j" in str(loops[2]["name"])
        assert "0 to 20" in str(loops[2]["body"])
        assert "l" in str(loops[3]["name"])
        assert "0 to 40" in str(loops[3]["body"])

    # axes are not consecutive
    def test_case_2():
        a, b, c = algo()
        s = hcl.create_schedule([a, b])
        s[c].reorder(c.axis[3], c.axis[0])
        ir = hcl.lower(s)
        loops = hcl_mlir.get_affine_loop_nests(s.top_func)[0]
        assert "l" in str(loops[0]["name"])
        assert "0 to 40" in str(loops[0]["body"])
        assert "j" in str(loops[1]["name"])
        assert "0 to 20" in str(loops[1]["body"])
        assert "k" in str(loops[2]["name"])
        assert "0 to 30" in str(loops[2]["body"])
        assert "i" in str(loops[3]["name"])
        assert "0 to 10" in str(loops[3]["body"])

    test_case_1()
    test_case_2()


def test_reorder_num_axis():
    hcl.init()
    a = hcl.placeholder((10, 20, 30, 40), name="a")
    b = hcl.placeholder((10, 20, 30, 40), name="b")
    c = hcl.compute(a.shape, lambda i, j, k, l: a[i, j, k, l] + b[i, j, k, l], name="c")

    s = hcl.create_schedule([a, b])
    s[c].reorder(2, 1)
    ir = hcl.lower(s)
    loops = hcl_mlir.get_affine_loop_nests(s.top_func)[0]
    assert "i" in str(loops[0]["name"])
    assert "0 to 10" in str(loops[0]["body"])
    assert "k" in str(loops[1]["name"])
    assert "0 to 30" in str(loops[1]["body"])
    assert "j" in str(loops[2]["name"])
    assert "0 to 20" in str(loops[2]["body"])
    assert "l" in str(loops[3]["name"])
    assert "0 to 40" in str(loops[3]["body"])


def test_split():
    def algo():
        hcl.init()
        a = hcl.placeholder((10, 20), name="a")
        b = hcl.placeholder((10, 20), name="b")
        c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")
        return a, b, c

    # without if condition
    def test_transform_mode_1():
        a, b, c = algo()
        s = hcl.create_schedule([a, b])
        s[c].split(c.axis[1], factor=4, mode="transform")
        ir = hcl.lower(s)
        loops = hcl_mlir.get_affine_loop_nests(s.top_func)[0]
        assert "i" in str(loops[0]["name"])
        assert "0 to 10" in str(loops[0]["body"])
        assert "j.outer" in str(loops[1]["name"])
        assert "0 to 5" in str(loops[1]["body"])
        assert "j.inner" in str(loops[2]["name"])
        assert "0 to 4" in str(loops[2]["body"])

    # with if condition
    def test_transform_mode_2():
        a, b, c = algo()
        s = hcl.create_schedule([a, b])
        s[c].split(c.axis[1], factor=3, mode="transform")
        ir = hcl.lower(s)
        loops = hcl_mlir.get_affine_loop_nests(s.top_func)[0]
        assert "i" in str(loops[0]["name"])
        assert "0 to 10" in str(loops[0]["body"])
        assert "j.outer" in str(loops[1]["name"])
        assert "0 to 7" in str(loops[1]["body"])
        assert "j.inner" in str(loops[2]["name"])
        assert "0 to min affine_map<(d0) -> (3, d0 * -3 + 20)>" in str(loops[2]["body"])

    # def test_annotate_mode():
    #     split_factor = 3
    #     s = hcl.create_schedule([a, b])
    #     s[c].split(c.axis[1], factor=split_factor, mode="annotate")
    #     split_hint_str = "split_factor="+str(split_factor)
    #     ir = hcl.lower(s)
    #     assert split_hint_str in str(ir)

    test_transform_mode_1()
    test_transform_mode_2()
    # test_annotate_mode()


def test_split_num_axis():
    hcl.init()
    a = hcl.placeholder((10, 20), name="a")
    b = hcl.placeholder((10, 20), name="b")
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")

    s = hcl.create_schedule([a, b])
    s[c].split(1, factor=4, mode="transform")
    ir = hcl.lower(s)
    loops = hcl_mlir.get_affine_loop_nests(s.top_func)[0]
    assert "i" in str(loops[0]["name"])
    assert "0 to 10" in str(loops[0]["body"])
    assert "j.outer" in str(loops[1]["name"])
    assert "0 to 5" in str(loops[1]["body"])
    assert "j.inner" in str(loops[2]["name"])
    assert "0 to 4" in str(loops[2]["body"])


def test_split_reorder():
    def algo():
        hcl.init()
        a = hcl.placeholder((10, 20), name="a")
        b = hcl.placeholder((10, 20), name="b")
        c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")
        return a, b, c

    def test_case_1():
        a, b, c = algo()
        s = hcl.create_schedule([a, b])
        xo, xi = s[c].split(c.axis[0], factor=2, mode="transform")
        yo, yi = s[c].split(c.axis[1], factor=5, mode="transform")
        s[c].reorder(yo, xo, yi, xi)
        ir = hcl.lower(s)
        loops = hcl_mlir.get_affine_loop_nests(s.top_func)[0]
        assert "j.outer" in str(loops[0]["name"])
        assert "0 to 4" in str(loops[0]["body"])
        assert "i.outer" in str(loops[1]["name"])
        assert "0 to 5" in str(loops[1]["body"])
        assert "j.inner" in str(loops[2]["name"])
        assert "0 to 5" in str(loops[2]["body"])
        assert "i.inner" in str(loops[3]["name"])
        assert "0 to 2" in str(loops[3]["body"])

    def test_case_2():
        a, b, c = algo()
        s = hcl.create_schedule([a, b])
        xo, xi = s[c].split(c.axis[0], factor=3, mode="transform")
        yo, yi = s[c].split(c.axis[1], factor=3, mode="transform")
        s[c].reorder(yo, yi, xo, xi)
        ir = hcl.lower(s)
        loops = hcl_mlir.get_affine_loop_nests(s.top_func)[0]
        assert "j.outer" in str(loops[0]["name"])
        assert "0 to 7" in str(loops[0]["body"])
        assert "j.inner" in str(loops[1]["name"])
        assert "0 to min affine_map<(d0) -> (3, d0 * -3 + 20)>" in str(loops[1]["body"])
        assert "i.outer" in str(loops[2]["name"])
        assert "0 to 4" in str(loops[2]["body"])
        assert "i.inner" in str(loops[3]["name"])
        assert "0 to min affine_map<(d0) -> (3, d0 * -3 + 10)>" in str(loops[3]["body"])

    test_case_1()
    test_case_2()


def test_split_reorder_num_axis():
    # note that this is not the recommanded way
    hcl.init()
    a = hcl.placeholder((10, 20), name="a")
    b = hcl.placeholder((10, 20), name="b")
    c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")

    s = hcl.create_schedule([a, b])
    xo, xi = s[c].split(0, factor=2, mode="transform")
    yo, yi = s[c].split(1, factor=5, mode="transform")
    s[c].reorder(yo, xo, yi, xi)
    ir = hcl.lower(s)
    loops = hcl_mlir.get_affine_loop_nests(s.top_func)[0]
    assert "j.outer" in str(loops[0]["name"])
    assert "0 to 4" in str(loops[0]["body"])
    assert "i.outer" in str(loops[1]["name"])
    assert "0 to 5" in str(loops[1]["body"])
    assert "j.inner" in str(loops[2]["name"])
    assert "0 to 5" in str(loops[2]["body"])
    assert "i.inner" in str(loops[3]["name"])
    assert "0 to 2" in str(loops[3]["body"])


def test_compute_at():
    def _build_kernel():
        hcl.init()
        A = hcl.placeholder((10, 20, 30), name="A")
        B = hcl.compute(A.shape, lambda i, j, m: A[i, j, m] * 2, name="B")
        C = hcl.compute(B.shape, lambda ii, jj, mm: B[ii, jj, mm] + 1, name="C")
        return A, B, C

    def _verify_build(sch):
        f = hcl.build(sch)
        a_np = np.random.randint(low=0, high=100, size=(10, 20, 30))
        a_hcl = hcl.asarray(a_np)
        c_hcl = hcl.asarray(np.zeros(a_np.shape), dtype=hcl.Int(32))
        f(a_hcl, c_hcl)
        c_np = a_np * 2 + 1
        np.testing.assert_allclose(c_np, c_hcl.asnumpy())

    def test_case_1():
        # axis 0
        A, B, C = _build_kernel()
        s0 = hcl.create_schedule([A, C])
        s0[B].compute_at(s0[C], C.axis[0])
        ir0 = hcl.lower(s0)
        loop = hcl_mlir.get_affine_loop_nests(s0.top_func)[0][0]["body"]
        assert "j" in str(loop.body.operations[0].attributes["loop_name"])
        assert "0 to 20" in str(loop.body.operations[0])
        assert "jj" in str(loop.body.operations[1].attributes["loop_name"])
        assert "0 to 20" in str(loop.body.operations[1])
        _verify_build(s0)
        # axis 1
        A, B, C = _build_kernel()
        s1 = hcl.create_schedule([A, C])
        s1[B].compute_at(s1[C], C.axis[1])
        ir1 = hcl.lower(s1)
        loop = hcl_mlir.get_affine_loop_nests(s1.top_func)[0][1]["body"]
        assert "m" in str(loop.body.operations[0].attributes["loop_name"])
        assert "0 to 30" in str(loop.body.operations[0])
        assert "mm" in str(loop.body.operations[1].attributes["loop_name"])
        assert "0 to 30" in str(loop.body.operations[1])
        _verify_build(s1)
        # axis 2
        A, B, C = _build_kernel()
        s2 = hcl.create_schedule([A, C])
        s2[B].compute_at(s2[C], C.axis[2])
        ir2 = hcl.lower(s2)
        loop = hcl_mlir.get_affine_loop_nests(s2.top_func)[0][2]["body"]
        assert "mm" in str(loop.attributes["loop_name"])
        assert "0 to 30" in str(loop)
        _verify_build(s2)

    def test_case_2():
        A, B, C = _build_kernel()
        s = hcl.create_schedule([A, C])
        s[B].compute_at(s[C], C.axis[2])
        s[C].fuse(C.axis[0], C.axis[1])
        ir = hcl.lower(s)
        loop = hcl_mlir.get_affine_loop_nests(s.top_func)[0][0]["body"]
        assert "ii_jj_fused" in str(loop.attributes["loop_name"])
        assert "0 to 200" in str(loop)
        _verify_build(s)

    def test_case_3():
        A, B, C = _build_kernel()
        s = hcl.create_schedule([A, C])
        s[B].compute_at(s[C], C.axis[2])
        s[C].split(C.axis[0], factor=3)
        s[C].split(C.axis[1], factor=3)
        ir = hcl.lower(s)
        loops = hcl_mlir.get_affine_loop_nests(s.top_func)[0]
        assert "ii.outer" in str(loops[0]["name"])
        assert "0 to 4" in str(loops[0]["body"])
        assert "ii.inner" in str(loops[1]["name"])
        assert "0 to min affine_map<(d0) -> (3, d0 * -3 + 10)>" in str(loops[1]["body"])
        assert "jj.outer" in str(loops[2]["name"])
        assert "0 to 7" in str(loops[2]["body"])
        assert "jj.inner" in str(loops[3]["name"])
        assert "0 to min affine_map<(d0) -> (3, d0 * -3 + 20)>" in str(loops[3]["body"])
        assert "mm" in str(loops[4]["name"])
        assert "0 to 30" in str(loops[4]["body"])
        _verify_build(s)

    # compute_at and reorder, compute at an axis that is not reordered
    # check both directions of reorder and compute_at
    def test_case_4():
        A, B, C = _build_kernel()
        s0 = hcl.create_schedule([A, C])
        s0[B].compute_at(s0[C], C.axis[2])
        s0[C].reorder(C.axis[1], C.axis[0])
        ir0 = hcl.lower(s0)
        loops = hcl_mlir.get_affine_loop_nests(s0.top_func)[0]
        assert "jj" in str(loops[0]["name"])
        assert "0 to 20" in str(loops[0]["body"])
        assert "ii" in str(loops[1]["name"])
        assert "0 to 10" in str(loops[1]["body"])
        assert "mm" in str(loops[2]["name"])
        assert "0 to 30" in str(loops[2]["body"])
        _verify_build(s0)

    # compute_at and reorder, compute at an axis that has been reordered
    # note that the results will be different
    def test_case_5():
        A, B, C = _build_kernel()
        s0 = hcl.create_schedule([A, C])
        s0[B].compute_at(s0[C], C.axis[1])
        s0[C].reorder(C.axis[1], C.axis[0])
        ir0 = hcl.lower(s0)
        loops = hcl_mlir.get_affine_loop_nests(s0.top_func)[0]
        assert "jj" in str(loops[0]["name"])
        assert "0 to 20" in str(loops[0]["body"])
        assert "ii" in str(loops[1]["name"])
        assert "0 to 10" in str(loops[1]["body"])
        assert "m" in str(loops[2]["name"])
        assert "0 to 30" in str(loops[2]["body"])
        assert "mm" in str(loops[1]["body"].body.operations[1].attributes["loop_name"])
        assert "0 to 30" in str(loops[1]["body"].body.operations[1])
        _verify_build(s0)

    def test_case_6():
        A, B, C = _build_kernel()
        s = hcl.create_schedule([A, C])
        s[B].compute_at(s[C], C.axis[2])
        yo, yi = s[C].split(C.axis[0], factor=3)
        xo, xi = s[C].split(C.axis[1], factor=3)
        s[C].reorder(yo, xo, yi, xi)
        ir = hcl.lower(s)
        loops = hcl_mlir.get_affine_loop_nests(s.top_func)[0]
        assert "ii.outer" in str(loops[0]["name"])
        assert "0 to 4" in str(loops[0]["body"])
        assert "jj.outer" in str(loops[1]["name"])
        assert "0 to 7" in str(loops[1]["body"])
        assert "ii.inner" in str(loops[2]["name"])
        assert "0 to min affine_map<(d0) -> (3, d0 * -3 + 10)>" in str(loops[2]["body"])
        assert "jj.inner" in str(loops[3]["name"])
        assert "0 to min affine_map<(d0) -> (3, d0 * -3 + 20)>" in str(loops[3]["body"])
        assert "mm" in str(loops[4]["name"])
        assert "0 to 30" in str(loops[4]["body"])
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
    loop_nests = hcl_mlir.get_affine_loop_nests(s.top_func)
    assert len(loop_nests) == 1
    loops = loop_nests[0]
    assert "jjj" in str(loops[1]["name"])
    assert "mmm" in str(loops[2]["name"])
    assert "m" in str(loops[3]["name"])
    f = hcl.build(s)
    a_np = np.random.randint(low=0, high=100, size=A.shape)
    a_hcl = hcl.asarray(a_np, dtype=hcl.Int(32))
    d_hcl = hcl.asarray(np.zeros(D.shape, dtype=np.int32), dtype=hcl.Int(32))
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
    loop_nests = hcl_mlir.get_affine_loop_nests(s.top_func)
    assert len(loop_nests) == 1
    loops = loop_nests[0]
    assert "jjj" in str(loops[1]["name"])
    assert "mmm" in str(loops[2]["name"])
    assert "m" in str(loops[3]["name"])
    f = hcl.build(s)
    a_np = np.random.randint(low=0, high=100, size=A.shape)
    a_hcl = hcl.asarray(a_np, dtype=hcl.Int(32))
    d_hcl = hcl.asarray(np.zeros(D.shape, dtype=np.int32), dtype=hcl.Int(32))
    f(a_hcl, d_hcl)
    d_np = (a_np * 2 + 1) % 3
    np.testing.assert_allclose(d_np, d_hcl.asnumpy())


def test_compute_at_with_reuse_1D():
    hcl.init()

    def _kernel():
        A = hcl.compute((10, 10), lambda y, x: x + y, "A")
        B = hcl.compute((10, 8), lambda y, x: A[y, x] + A[y, x + 1] + A[y, x + 2], "B")
        return B

    s = hcl.create_schedule([], _kernel)
    A = _kernel.A
    B = _kernel.B
    s[A].compute_at(s[B], B.axis[1])
    ir = hcl.lower(s)
    loops = hcl_mlir.get_affine_loop_nests(s.top_func)
    assert len(loops) == 1
    f = hcl.build(s)
    a_np = np.fromfunction(lambda i, j: i + j, A.shape, dtype="int")
    b_np = np.zeros(B.shape, dtype="int")
    c_np = np.zeros(B.shape, dtype="int")
    for y in range(0, 10):
        for x in range(0, 8):
            c_np[y][x] = a_np[y][x] + a_np[y][x + 1] + a_np[y][x + 2]
    b_hcl = hcl.asarray(b_np)
    f(b_hcl)
    np.testing.assert_array_equal(c_np, b_hcl.asnumpy())


def test_compute_at_with_reuse_2D():
    hcl.init()

    def _kernel():
        A = hcl.compute((10, 10), lambda y, x: x + y, name="A", dtype=hcl.Int(32))
        B = hcl.compute(
            (8, 8),
            lambda y, x: A[y, x] + A[y + 1, x + 1] + A[y + 2, x + 2],
            name="B",
            dtype=hcl.Int(32),
        )
        return B

    s = hcl.create_schedule([], _kernel)
    A = _kernel.A
    B = _kernel.B
    s[A].compute_at(s[B], B.axis[1])
    ir = hcl.lower(s)
    loops = hcl_mlir.get_affine_loop_nests(s.top_func)
    assert len(loops) == 1
    f = hcl.build(s)
    a_np = np.fromfunction(lambda i, j: i + j, A.shape, dtype="int")
    b_np = np.zeros(B.shape, dtype="int")
    c_np = np.zeros(B.shape, dtype="int")
    for y in range(0, 8):
        for x in range(0, 8):
            c_np[y][x] = a_np[y][x] + a_np[y + 1][x + 1] + a_np[y + 2][x + 2]
    b_hcl = hcl.asarray(b_np)
    f(b_hcl)
    np.testing.assert_array_equal(c_np, b_hcl.asnumpy())


def test_compute_at_with_reuse_2D_complex():
    hcl.init()

    def _kernel():
        A = hcl.compute((10, 10), lambda y, x: x + y, "A")
        B = hcl.compute(
            (8, 8), lambda y, x: A[y, x] + A[y + 1, x + 1] + A[y + 2, x + 2], "B"
        )
        return B

    s = hcl.create_schedule([], _kernel)
    A = _kernel.A
    B = _kernel.B
    s[A].compute_at(s[B], B.axis[1])
    s[B].split(B.axis[1], 4)
    ir = hcl.lower(s)
    loops = hcl_mlir.get_affine_loop_nests(s.top_func)[0]
    assert "y" in str(loops[0]["name"])
    assert "0 to 8" in str(loops[0]["body"])
    assert "x_0.outer" in str(loops[1]["name"])
    assert "0 to 2" in str(loops[1]["body"])
    assert "x_0.inner" in str(loops[2]["name"])
    assert "0 to 4" in str(loops[2]["body"])
    f = hcl.build(s)
    a_np = np.fromfunction(lambda i, j: i + j, A.shape, dtype="int")
    b_np = np.zeros(B.shape, dtype="int")
    c_np = np.zeros(B.shape, dtype="int")
    for y in range(0, 8):
        for x in range(0, 8):
            c_np[y][x] = a_np[y][x] + a_np[y + 1][x + 1] + a_np[y + 2][x + 2]
    b_hcl = hcl.asarray(b_np)
    f(b_hcl)
    np.testing.assert_array_equal(c_np, b_hcl.asnumpy())


def test_compute_at_no_dep():
    hcl.init()
    A = hcl.compute((10, 10), lambda y, x: y + x, "A")
    B = hcl.compute((10, 10), lambda y, x: y - x, "B")
    s = hcl.create_schedule([A, B])
    s[A].compute_at(s[B], B.axis[1])
    f = hcl.build(s)
    a_hcl = hcl.asarray(np.zeros(A.shape, dtype="int"))
    b_hcl = hcl.asarray(np.zeros(B.shape, dtype="int"))
    f(a_hcl, b_hcl)
    a_np = np.fromfunction(lambda i, j: i + j, A.shape, dtype="int")
    b_np = np.fromfunction(lambda i, j: i - j, B.shape, dtype="int")
    np.testing.assert_array_equal(a_np, a_hcl.asnumpy())
    np.testing.assert_array_equal(b_np, b_hcl.asnumpy())


def test_compute_at_no_dep_diff_shape_smaller():
    hcl.init()
    A = hcl.compute((8, 8), lambda y, x: y + x, "A")
    B = hcl.compute((10, 10), lambda y, x: y - x, "B")
    s = hcl.create_schedule([A, B])
    s[A].compute_at(s[B], B.axis[1])
    f = hcl.build(s)
    a_hcl = hcl.asarray(np.zeros(A.shape, dtype="int"))
    b_hcl = hcl.asarray(np.zeros(B.shape, dtype="int"))
    f(a_hcl, b_hcl)
    a_np = np.fromfunction(lambda i, j: i + j, A.shape, dtype="int")
    b_np = np.fromfunction(lambda i, j: i - j, B.shape, dtype="int")
    np.testing.assert_array_equal(a_np, a_hcl.asnumpy())
    np.testing.assert_array_equal(b_np, b_hcl.asnumpy())


"""
Note: 
    In Halide-based HeteroCL, A will be capped to (10, 10).
    In MLIR-based HeteroCL, A will not be capped. 
    i.e., compute_at does not change the algorithm.
"""


def test_compute_at_no_dep_diff_shape_larger():
    hcl.init()
    A = hcl.compute((12, 12), lambda y, x: y + x, "A")
    B = hcl.compute((10, 10), lambda y, x: y - x, "B")
    s = hcl.create_schedule([A, B])
    s[A].compute_at(s[B], B.axis[1])
    f = hcl.build(s)
    a_hcl = hcl.asarray(np.zeros(A.shape, dtype="int"), dtype=hcl.Int(32))
    b_hcl = hcl.asarray(np.zeros(B.shape, dtype="int"), dtype=hcl.Int(32))
    f(a_hcl, b_hcl)
    a_np = np.fromfunction(lambda i, j: i + j, A.shape, dtype="int")
    b_np = np.fromfunction(lambda i, j: i - j, B.shape, dtype="int")
    np.testing.assert_array_equal(a_np, a_hcl.asnumpy())
    np.testing.assert_array_equal(b_np, b_hcl.asnumpy())


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
    a_hcl = hcl.asarray(a_np, dtype=hcl.Int(32))
    b_hcl = hcl.asarray(b_np, dtype=hcl.Int(32))
    f(a_hcl, b_hcl)
    d_np = np.sum(a_np, axis=1)
    np.testing.assert_array_equal(d_np, b_hcl.asnumpy())
