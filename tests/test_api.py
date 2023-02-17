# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np
import pytest


def test_schedule_no_return():
    hcl.init()
    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    def algorithm(A, B):
        hcl.update(B, lambda x: A[x] + 1)

    s = hcl.create_schedule([A, B], algorithm)
    f = hcl.build(s)

    _A = hcl.asarray(np.random.randint(100, size=(10,)), dtype=hcl.Int(32))
    _B = hcl.asarray(np.zeros(10), dtype=hcl.Int(32))

    f(_A, _B)

    _A = _A.asnumpy()
    _B = _B.asnumpy()

    for i in range(10):
        assert _B[i] == _A[i] + 1


def test_schedule_return():
    hcl.init()
    A = hcl.placeholder((10,))

    def algorithm(A):
        return hcl.compute(A.shape, lambda x: A[x] + 1)

    s = hcl.create_schedule([A], algorithm)
    f = hcl.build(s)

    _A = hcl.asarray(np.random.randint(100, size=(10,)), dtype=hcl.Int(32))
    _B = hcl.asarray(np.zeros(10), dtype=hcl.Int(32))

    f(_A, _B)

    _A = _A.asnumpy()
    _B = _B.asnumpy()

    for i in range(10):
        assert _B[i] == _A[i] + 1


def test_schedule_return_multi():
    hcl.init()
    A = hcl.placeholder((10,))

    def algorithm(A):
        B = hcl.compute(A.shape, lambda x: A[x] + 1)
        C = hcl.compute(A.shape, lambda x: A[x] + 2)
        return B, C

    s = hcl.create_schedule([A], algorithm)
    f = hcl.build(s)

    _A = hcl.asarray(np.random.randint(100, size=(10,)), dtype=hcl.Int(32))
    _B = hcl.asarray(np.zeros(10), dtype=hcl.Int(32))
    _C = hcl.asarray(np.zeros(10), dtype=hcl.Int(32))

    f(_A, _B, _C)

    _A = _A.asnumpy()
    _B = _B.asnumpy()
    _C = _C.asnumpy()

    for i in range(10):
        assert _B[i] == _A[i] + 1
        assert _C[i] == _A[i] + 2


def test_resize():
    hcl.init()

    def algorithm(A):
        return hcl.compute(A.shape, lambda x: A[x] + 1, "B")

    A = hcl.placeholder((10,), dtype=hcl.UInt(32))

    scheme = hcl.create_scheme([A], algorithm)
    scheme.downsize(algorithm.B, hcl.UInt(2))
    s = hcl.create_schedule_from_scheme(scheme)
    f = hcl.build(s)
    a = np.random.randint(100, size=(10,))
    _A = hcl.asarray(a, dtype=hcl.UInt(32))
    _B = hcl.asarray(np.zeros(10), dtype=hcl.UInt(2))

    f(_A, _B)

    _A = _A.asnumpy()
    _B = _B.asnumpy()

    for i in range(10):
        assert _B[i] == (a[i] + 1) % 4


def test_select():
    hcl.init(hcl.Float())
    A = hcl.placeholder((10,))
    B = hcl.compute(A.shape, lambda x: hcl.select(A[x] > 0.5, A[x], 0.0))
    s = hcl.create_schedule([A, B])
    f = hcl.build(s)

    np_A = np.random.rand(10)
    np_B = np.zeros(10)
    np_C = np.zeros(10)

    for i in range(0, 10):
        np_C[i] = np_A[i] if np_A[i] > 0.5 else 0

    hcl_A = hcl.asarray(np_A, dtype=hcl.Float(32))
    hcl_B = hcl.asarray(np_B, dtype=hcl.Float(32))

    f(hcl_A, hcl_B)

    np_B = hcl_B.asnumpy()

    assert np.allclose(np_B, np_C)


def test_bitwise_and():
    hcl.init(hcl.UInt(8))

    N = 100
    A = hcl.placeholder((N, N))
    B = hcl.placeholder((N, N))

    def kernel(A, B):
        return hcl.compute(A.shape, lambda x, y: A[x, y] & B[x, y])

    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    a = np.random.randint(0, 255, (N, N))
    b = np.random.randint(0, 255, (N, N))
    c = np.zeros((N, N))
    g = a & b

    hcl_a = hcl.asarray(a)
    hcl_b = hcl.asarray(b)
    hcl_c = hcl.asarray(c)
    f(hcl_a, hcl_b, hcl_c)
    assert np.array_equal(hcl_c.asnumpy(), g)


def test_bitwise_or():
    hcl.init(hcl.UInt(8))

    N = 100
    A = hcl.placeholder((N, N))
    B = hcl.placeholder((N, N))

    def kernel(A, B):
        return hcl.compute(A.shape, lambda x, y: A[x, y] | B[x, y])

    s = hcl.create_schedule([A, B], kernel)
    f = hcl.build(s)

    a = np.random.randint(0, 255, (N, N))
    b = np.random.randint(0, 255, (N, N))
    c = np.zeros((N, N))
    g = a | b

    hcl_a = hcl.asarray(a)
    hcl_b = hcl.asarray(b)
    hcl_c = hcl.asarray(c)
    f(hcl_a, hcl_b, hcl_c)
    assert np.array_equal(hcl_c.asnumpy(), g)


def test_tensor_slice_shape():
    A = hcl.placeholder((3, 4, 5))

    assert A.shape == (3, 4, 5)
    assert A[0].shape == (4, 5)
    assert A[0][1].shape == (5,)


def test_str_fmt_asarray():
    shape = (3, 4, 5)
    A = hcl.placeholder(shape, dtype=hcl.Int(10))
    B = hcl.placeholder(shape, dtype=hcl.Float(32))
    C = hcl.placeholder(shape, dtype=hcl.Fixed(5, 2))
    D = hcl.placeholder(shape, dtype=hcl.UInt(8))
    E = hcl.placeholder(shape, dtype=hcl.UFixed(8, 4))

    def kernel(A, B, C, D, E):
        A_ret = hcl.compute(A.shape, lambda *args: A[args], "A_ret", dtype=A.dtype)
        B_ret = hcl.compute(B.shape, lambda *args: B[args], "B_ret", dtype=B.dtype)
        C_ret = hcl.compute(C.shape, lambda *args: C[args], "C_ret", dtype=C.dtype)
        D_ret = hcl.compute(D.shape, lambda *args: D[args], "D_ret", dtype=D.dtype)
        E_ret = hcl.compute(E.shape, lambda *args: E[args], "E_ret", dtype=E.dtype)
        return A_ret, B_ret, C_ret, D_ret, E_ret

    s = hcl.create_schedule([A, B, C, D, E], kernel)
    f = hcl.build(s)
    np_A = np.random.randint(-100, 100, shape)
    np_B = np.random.rand(*shape)
    np_C = np.random.rand(*shape)
    np_D = np.random.randint(0, 255, shape)
    np_E = np.random.rand(*shape)

    hcl_A = hcl.asarray(np_A, dtype="int10")
    hcl_B = hcl.asarray(np_B, dtype="float32")
    hcl_C = hcl.asarray(np_C, dtype="fixed5_2")
    hcl_D = hcl.asarray(np_D, dtype="uint8")
    hcl_E = hcl.asarray(np_E, dtype="ufixed8_4")

    hcl_A_ret = hcl.asarray(np.zeros(shape), dtype="int10")
    hcl_B_ret = hcl.asarray(np.zeros(shape), dtype="float32")
    hcl_C_ret = hcl.asarray(np.zeros(shape), dtype="fixed5_2")
    hcl_D_ret = hcl.asarray(np.zeros(shape), dtype="uint8")
    hcl_E_ret = hcl.asarray(np.zeros(shape), dtype="ufixed8_4")

    f(
        hcl_A,
        hcl_B,
        hcl_C,
        hcl_D,
        hcl_E,
        hcl_A_ret,
        hcl_B_ret,
        hcl_C_ret,
        hcl_D_ret,
        hcl_E_ret,
    )

    ret_A = hcl_A_ret.asnumpy()
    ret_B = hcl_B_ret.asnumpy()
    ret_C = hcl_C_ret.asnumpy()
    ret_D = hcl_D_ret.asnumpy()
    ret_E = hcl_E_ret.asnumpy()

    golden_A = hcl.cast_np(np_A, "int10")
    golden_B = hcl.cast_np(np_B, "float32")
    golden_C = hcl.cast_np(np_C, "fixed5_2")
    golden_D = hcl.cast_np(np_D, "uint8")
    golden_E = hcl.cast_np(np_E, "ufixed8_4")

    assert np.array_equal(ret_A, golden_A)
    assert np.array_equal(ret_B, golden_B)
    assert np.array_equal(ret_C, golden_C)
    assert np.array_equal(ret_D, golden_D)
    assert np.array_equal(ret_E, golden_E)
