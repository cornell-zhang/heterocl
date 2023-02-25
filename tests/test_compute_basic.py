# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np
import pytest


def test_rhs_binaryop():
    hcl.init()

    def kernel():
        v = hcl.scalar(5, "v")
        res = hcl.compute((11,), lambda i: 0, dtype=hcl.Int(32))
        res[0] = 1 + v.v
        res[1] = 1 - v.v
        res[2] = 1 * v.v
        res[3] = 52 / v.v
        res[4] = 6 // v.v
        res[5] = 6 % v.v
        res[6] = 1 << v.v
        res[7] = 64 >> v.v
        res[8] = 1 & v.v
        res[9] = 1 | v.v
        res[10] = 1 ^ v.v
        return res

    s = hcl.create_schedule([], kernel)
    f = hcl.build(s)
    hcl_res = hcl.asarray(np.zeros((11,), dtype=np.int32))
    f(hcl_res)
    np_res = np.zeros((11,), dtype=np.int32)
    np_res[0] = 1 + 5
    np_res[1] = 1 - 5
    np_res[2] = 1 * 5
    np_res[3] = 52 / 5
    np_res[4] = 6 // 5
    np_res[5] = 6 % 5
    np_res[6] = 1 << 5
    np_res[7] = 64 >> 5
    np_res[8] = 1 & 5
    np_res[9] = 1 | 5
    np_res[10] = 1 ^ 5
    assert np.array_equal(hcl_res.asnumpy(), np_res)


def _test_kernel(kernel):
    A = hcl.placeholder((10,))
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(10)

    # TODO(Niansong): warn user of mismatched np dtype and hcldtype
    #  hcl.asarray(np_A) would give us incorrect results
    # because of incorrect default dtype (f32)
    hcl_A = hcl.asarray(np_A, dtype=hcl.Int(32))
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()

    for i in range(0, 10):
        assert ret_B[i] == np_A[i] + 1


def test_fcompute_basic():
    def kernel(A):
        return hcl.compute(A.shape, lambda x: A[x] + 1)

    _test_kernel(kernel)


def test_fcompute_function_wrapper():
    def kernel(A):
        def foo(x):
            return x + 1

        return hcl.compute(A.shape, lambda x: foo(A[x]))

    _test_kernel(kernel)


def test_fcompute_wrap_more():
    def kernel(A):
        def foo(x):
            return A[x] + 1

        return hcl.compute(A.shape, lambda x: foo(x))

    _test_kernel(kernel)


def test_fcompute_no_lambda():
    def kernel(A):
        def foo(x):
            return A[x] + 1

        return hcl.compute(A.shape, foo)

    _test_kernel(kernel)


def test_fcompute_imperative_return():
    def kernel(A):
        def foo(x):
            hcl.return_(A[x] + 1)

        return hcl.compute(A.shape, foo)

    _test_kernel(kernel)


def test_fcompute_imperative_function():
    def kernel(A):
        @hcl.def_([A.shape, ()])
        def foo(A, x):
            hcl.return_(A[x] + 1)

        return hcl.compute(A.shape, lambda x: foo(A, x))

    _test_kernel(kernel)


def test_fcompute_nested():
    def kernel(A):
        def foo(A, x):
            B = hcl.compute(A.shape, lambda y: A[y] + 1)
            return B[x]

        return hcl.compute(A.shape, lambda x: foo(A, x))

    _test_kernel(kernel)


def test_fcompute_nested_imperative():
    def kernel(A):
        def foo(A, x):
            B = hcl.compute(A.shape, lambda y: A[y] + 1)
            hcl.return_(B[x])

        return hcl.compute(A.shape, lambda x: foo(A, x))

    _test_kernel(kernel)


def _test_fcompute_multiple_return():
    def kernel(A):
        def foo(x):
            with hcl.if_(A[x] > 5):
                hcl.return_(x)
            with hcl.else_():
                hcl.return_(0)

        return hcl.compute(A.shape, foo)

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(10)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()

    for i in range(0, 10):
        if np_A[i] > 5:
            assert ret_B[i] == i
        else:
            assert ret_B[i] == 0


def test_fcompute_multiple_return():
    with pytest.raises(Exception):
        _test_fcompute_multiple_return()


def _test_fcompute_multiple_return_multi_dim():
    def kernel(A):
        def foo(x, y, z):
            with hcl.if_(A[x, y, z] > 5):
                hcl.return_(x)
            with hcl.else_():
                hcl.return_(0)

        return hcl.compute(A.shape, foo)

    A = hcl.placeholder((10, 10, 10))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 10, 10))
    np_B = np.zeros((10, 10, 10))

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()

    for i in range(0, 10):
        for j in range(0, 10):
            for k in range(0, 10):
                if np_A[i][j][k] > 5:
                    assert ret_B[i][j][k] == i
                else:
                    assert ret_B[i][j][k] == 0


def test_fcompute_multiple_return_multi_dim():
    with pytest.raises(Exception):
        _test_fcompute_multiple_return_multi_dim()


def test_update():
    def kernel(A, B):
        hcl.update(B, lambda x: A[x] + 1)

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(10)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()

    for i in range(0, 10):
        assert ret_B[i] == np_A[i] + 1


def test_copy():
    hcl.init()

    np_A = np.random.randint(10, size=(10, 10, 10))
    py_A = np_A.tolist()

    def kernel():
        cp1 = hcl.operation.copy(np_A)
        cp2 = hcl.operation.copy(py_A)
        return hcl.compute(np_A.shape, lambda *x: cp1[x] + cp2[x])

    O = hcl.placeholder(np_A.shape)
    s = hcl.create_schedule([], kernel)
    f = hcl.build(s)

    np_O = np.zeros(np_A.shape)
    hcl_O = hcl.asarray(np_O, dtype=hcl.Int(32))

    f(hcl_O)

    assert np.array_equal(hcl_O.asnumpy(), np_A * 2)


def test_mutate_basic():
    def kernel(A, B):
        def foo(x):
            B[x] = A[x] + 1

        hcl.mutate(A.shape, foo)

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(10)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()

    for i in range(0, 10):
        assert ret_B[i] == np_A[i] + 1


def test_mutate_complex():
    def kernel(A, B):
        def foo(x):
            with hcl.for_(0, 10) as y:
                with hcl.if_(A[x][y] > 5):
                    B[x] += 1

        hcl.mutate((10,), foo)

    A = hcl.placeholder((10, 10))
    B = hcl.placeholder((10,))
    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 10))
    np_B = np.zeros((10,))

    gold_B = []
    for i in range(0, 10):
        gold_B.append(len([x for x in np_A[i] if x > 5]))

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()

    for i in range(0, 10):
        assert ret_B[i] == gold_B[i]


def test_const_tensor_int():
    def test_kernel(dtype, size):
        hcl.init(dtype)

        np_A = np.random.randint(-15, 15, size=size, dtype=np.int32)
        py_A = np_A.tolist()

        def kernel():
            cp1 = hcl.const_tensor(np_A, dtype=dtype)
            cp2 = hcl.const_tensor(py_A, dtype=dtype)
            return hcl.compute(np_A.shape, lambda *x: cp1[x] + cp2[x])

        O = hcl.placeholder(np_A.shape)
        s = hcl.create_schedule([], kernel)
        f = hcl.build(s)
        # Test the generated bitwidth is the same as what user specified
        from hcl_mlir.dialects import memref as memref_d
        from hcl_mlir.ir import MemRefType, IntegerType, TypeAttr

        for op in s.module.body.operations:
            if isinstance(op, memref_d.GlobalOp):
                assert (
                    IntegerType(
                        MemRefType(TypeAttr(op.attributes["type"]).value).element_type
                    ).width
                    == dtype.bits
                )

        np_O = np.zeros(np_A.shape)
        hcl_O = hcl.asarray(np_O, dtype=dtype)

        f(hcl_O)

        assert np.array_equal(hcl_O.asnumpy(), np_A * 2)

    for i in range(0, 5):
        bit = np.random.randint(6, 60)
        test_kernel(hcl.Int(bit), (8, 8))
        test_kernel(hcl.Int(bit), (20, 20, 3))


def test_const_tensor_int_corner_case():
    def test_kernel(dtype, size):
        hcl.init(dtype)

        # 2^5 - 1 = 31
        np_A = np.random.randint(-15, 15, size=size, dtype=np.int32)
        py_A = np_A.tolist()

        def kernel():
            cp1 = hcl.const_tensor(np_A, dtype=dtype)
            cp2 = hcl.const_tensor(py_A, dtype=dtype)
            return hcl.compute(np_A.shape, lambda *x: cp1[x] + cp2[x])

        O = hcl.placeholder(np_A.shape)
        s = hcl.create_schedule([], kernel)
        f = hcl.build(s)
        np_O = np.zeros(np_A.shape)
        hcl_O = hcl.asarray(np_O, dtype=dtype)

        f(hcl_O)

        assert np.array_equal(hcl_O.asnumpy(), np_A * 2)

    test_kernel(hcl.Int(6), (8, 8))


def test_const_tensor_uint():
    def test_kernel(dtype, size):
        hcl.init(dtype)

        np_A = np.random.randint(32, size=size, dtype=np.int32)
        py_A = np_A.tolist()

        def kernel():
            cp1 = hcl.const_tensor(np_A, dtype=dtype)
            cp2 = hcl.const_tensor(py_A, dtype=dtype)
            return hcl.compute(np_A.shape, lambda *x: cp1[x] + cp2[x])

        O = hcl.placeholder(np_A.shape)
        s = hcl.create_schedule([], kernel)
        f = hcl.build(s)

        np_O = np.zeros(np_A.shape)
        hcl_O = hcl.asarray(np_O, dtype=dtype)

        f(hcl_O)

        assert np.array_equal(hcl_O.asnumpy(), np_A * 2)

    for i in range(0, 5):
        bit = np.random.randint(6, 60)
        test_kernel(hcl.UInt(bit), (8, 8))
        test_kernel(hcl.UInt(bit), (20, 20, 3))


def test_const_tensor_float():
    def test_kernel(dtype, size):
        hcl.init(dtype)

        np_A = np.random.rand(*size)
        py_A = np_A.tolist()

        def kernel():
            cp1 = hcl.const_tensor(np_A)
            cp2 = hcl.const_tensor(py_A)
            return hcl.compute(
                np_A.shape, lambda *x: cp1[x] + cp2[x], dtype=hcl.Float()
            )

        O = hcl.placeholder(np_A.shape)
        s = hcl.create_schedule([], kernel)
        f = hcl.build(s)

        np_O = np.zeros(np_A.shape)
        hcl_O = hcl.asarray(np_O, dtype=hcl.Float())

        f(hcl_O)

        np_A = hcl.cast_np(np_A, dtype)
        assert np.allclose(hcl_O.asnumpy(), np_A * 2, 1, 1e-5)

    test_kernel(hcl.Float(), (8, 8))
    test_kernel(hcl.Float(), (20, 20, 3))


def test_const_tensor_fixed():
    def test_kernel(dtype, size):
        hcl.init(dtype)

        np_A = np.random.rand(*size)
        py_A = np_A.tolist()

        def kernel():
            cp1 = hcl.const_tensor(np_A)
            cp2 = hcl.const_tensor(py_A)
            return hcl.compute(
                np_A.shape, lambda *x: cp1[x] + cp2[x], dtype=hcl.Float()
            )

        A = hcl.placeholder(np_A.shape)
        s = hcl.create_schedule([], kernel)
        f = hcl.build(s)

        np_O = np.zeros(np_A.shape)
        hcl_O = hcl.asarray(np_O, dtype=hcl.Float())

        f(hcl_O)

        np_A = hcl.cast_np(np_A, dtype)
        assert np.allclose(hcl_O.asnumpy(), np_A * 2, 1, 1e-5)

    for i in range(0, 5):
        bit = np.random.randint(10, 60)
        test_kernel(hcl.Fixed(bit, 4), (8, 8))
        test_kernel(hcl.UFixed(bit, 4), (8, 8))
        test_kernel(hcl.Fixed(bit, 4), (20, 20, 3))
        test_kernel(hcl.UFixed(bit, 4), (20, 20, 3))


def test_double_mutate_call():
    hcl.init()

    def kernel():
        n = 8192
        nh = n // 2
        _arf = hcl.compute((10, 10, n), lambda *_: 0, "arf", dtype="uint32")
        ain = hcl.compute((n,), lambda *_: 0, "ain", dtype="uint32")
        idx1 = hcl.scalar(0, "idx1", dtype="uint4")
        idx2 = hcl.scalar(0, "idx2", dtype="uint4")
        #
        arf = _arf[idx1.v][idx2.v]

        def outer(i):
            # actually any expression here will cause the error

            i64 = i * 64 + 199

            def inner(j, src, soffs):
                ain[j] = src[soffs + j]

            hcl.mutate((64,), lambda j: inner(j, arf, i64), "inner1")
            hcl.mutate((64,), lambda j: inner(j, arf, i64), "inner2")

        hcl.mutate((n // 128,), lambda i: outer(i), "outer")
        #
        r = hcl.compute((2,), lambda i: ain[0], dtype=hcl.UInt(32))
        return r

    s = hcl.create_schedule([], kernel)
    hcl.lower(s)


def test_expr_as_ub():
    hcl.init()

    def kernel():
        n = 8192
        nh = n // 2
        src = hcl.compute((n,), lambda *_: 0, "src", dtype="uint32")
        dst = hcl.compute((n,), lambda *_: 0, "dst", dtype="uint32")
        #
        v = hcl.scalar(0, "v", dtype="uint128")
        count = hcl.scalar(1, "count", dtype="uint32").v
        factor = 4
        #
        src_off = hcl.scalar(0, "src_off", dtype="uint32").v
        dst_off = hcl.scalar(0, "dst_off", dtype="uint32").v

        def check(cond, msg, args):
            _cond = hcl.scalar(cond, "assert_cond", dtype="uint1")
            with hcl.if_(_cond.v == 0):
                strfmt = "\n\nAssertion failed : " + msg + "\n\n"
                hcl.print(args, strfmt)

        check(
            src_off + count * factor <= src.shape[0],
            f"copyN: {src.name} index (%d) > size ({src.shape[0]})\n",
            (src_off + count * factor),
        )

        def funA(i):
            dst[i] = 0

        hcl.mutate((count,), funA)
        #
        r = hcl.compute((2,), lambda i: 0, dtype=hcl.UInt(32))
        return r

    s = hcl.create_schedule([], kernel)
    hcl.lower(s)
