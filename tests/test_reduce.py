# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np

import heterocl as hcl
import numpy as np


def test_reduce_basic():
    hcl.init()

    def kernel(A):
        my_sum = hcl.reducer(0, lambda x, y: x + y)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute((1,), lambda x: my_sum(A[r], axis=r))

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(1)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A)
    assert ret_B[0] == golden_B


def test_reduce_cond():
    def kernel(A):
        my_sum = hcl.reducer(0, lambda x, y: x + y)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute((1,), lambda x: my_sum(A[r], axis=r, where=A[r] > 5))

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(1)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A[np.where(np_A > 5)])
    assert ret_B[0] == golden_B


def test_reduce_dtype():
    def kernel(A):
        # my_sum will perform integer reduction
        my_sum = hcl.reducer(0, lambda x, y: x + y)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute(
            (1,), lambda x: my_sum(A[r], axis=r, dtype=hcl.Float()), dtype=hcl.Float()
        )

    A = hcl.placeholder((10,), dtype=hcl.Float())
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.rand(10)
    np_B = np.zeros(1)

    hcl_A = hcl.asarray(np_A, dtype=hcl.Float())
    hcl_B = hcl.asarray(np_B, dtype=hcl.Float())

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A)
    assert np.isclose(ret_B[0], golden_B)


def test_reduce_dtype_2():
    def kernel(A):
        my_sum = hcl.reducer(0, lambda x, y: x + y)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute((1,), lambda x: my_sum(A[r], axis=r, dtype=hcl.UInt(2)))

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(1)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A)
    assert ret_B[0] == golden_B % 4


def test_reduce_different_init():
    def kernel(a, A):
        my_sum = hcl.reducer(a.v, lambda x, y: x + y)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute((1,), lambda x: my_sum(A[r], axis=r))

    a = hcl.placeholder(())
    A = hcl.placeholder((10,))
    s = hcl.create_schedule([a, A], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(1)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(10, hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A)
    assert ret_B[0] == golden_B + 10


def test_reduce_2D():
    def kernel(A):
        my_sum = hcl.reducer(0, lambda x, y: x + y)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute((10,), lambda x: my_sum(A[r, x], axis=r))

    A = hcl.placeholder((10, 10))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 10))
    np_B = np.zeros(10)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A, axis=0)
    assert np.array_equal(ret_B, golden_B)


def test_reduce_2D_2():
    def kernel(A):
        my_sum = hcl.reducer(0, lambda x, y: x + y)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute((1, 10), lambda x, y: my_sum(A[r, y], axis=r))

    A = hcl.placeholder((10, 10))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 10))
    np_B = np.zeros((1, 10))

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A, axis=0)
    assert np.array_equal(ret_B[0], golden_B)


def test_reduce_multi_axes():
    def kernel(A):
        my_sum = hcl.reducer(0, lambda x, y: x + y)
        r1 = hcl.reduce_axis(0, 10)
        r2 = hcl.reduce_axis(0, 10)
        return hcl.compute((1,), lambda x: my_sum(A[r1, r2], axis=[r1, r2]))

    A = hcl.placeholder((10, 10))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10, 10))
    np_B = np.zeros(1)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = np.sum(np_A)
    assert ret_B[0] == golden_B


def test_reduce_complex_reducer():
    def kernel(A):
        def reducer_body(x, y):
            res = hcl.scalar(0, "res")
            with hcl.if_(x > 5):
                res.v = y + 1
            with hcl.else_():
                res.v = y + 2
            return res.v

        my_sum = hcl.reducer(0, reducer_body)
        r = hcl.reduce_axis(0, 10)
        return hcl.compute((1,), lambda x: my_sum(A[r], axis=r))

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=(10,))
    np_B = np.zeros(1)

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))

    f(hcl_A, hcl_B)

    ret_B = hcl_B.asnumpy()
    golden_B = 10 + len(np_A[np.where(np_A <= 5)])
    assert ret_B[0] == golden_B


def _test_maxpool(in_shape, out_shape, stride, kernel, test_dtype):
    def max_pool(data):
        h = hcl.reduce_axis(0, kernel)
        w = hcl.reduce_axis(0, kernel)
        return hcl.compute(
            out_shape,
            lambda hh, ww: hcl.max(
                data[stride * hh + h, stride * ww + w], axis=[h, w], dtype=test_dtype
            ),
            name="max_pool",
            dtype=test_dtype,
        )

    A = hcl.placeholder(in_shape, "A", dtype=test_dtype)
    s = hcl.create_schedule([A], max_pool)
    f = hcl.build(s)
    np_a = np.random.randint(0, 10, size=in_shape)
    a = hcl.asarray(np_a, dtype=test_dtype)
    b = hcl.asarray(np.zeros(out_shape), dtype=test_dtype)
    f(a, b)
    np_b = b.asnumpy()
    b_golden = np.zeros(out_shape)
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            b_golden[i, j] = np.max(
                np_a[i * stride : i * stride + kernel, j * stride : j * stride + kernel]
            )
    assert np.allclose(np_b, b_golden)


def test_maxpool():
    in_shape = (2, 8)
    out_shape = (1, 4)
    stride = 2
    kernel = 2

    test_dtypes = [hcl.Int(10), hcl.Int(8), hcl.Int(6), hcl.Fixed(12, 6)]

    for test_dtype in test_dtypes:
        _test_maxpool(in_shape, out_shape, stride, kernel, test_dtype)


def _test_meanpool(in_shape, out_shape, stride, kernel, test_dtype):
    def mean_pool(data):
        h = hcl.reduce_axis(0, kernel)
        w = hcl.reduce_axis(0, kernel)
        return hcl.compute(
            out_shape,
            lambda hh, ww: (
                hcl.sum(
                    data[stride * hh + h, stride * ww + w],
                    axis=[h, w],
                    dtype=test_dtype,
                )
            )
            / (kernel * kernel),
            name="mean_pool",
            dtype=test_dtype,
        )

    # 0,0 0,1 0,2 0,3 0,4 0,5 0,6 0,7
    # 1,0 1,1 1,2 1,3 1,4 1,5 1,6 1,7

    A = hcl.placeholder(in_shape, "A", dtype=test_dtype)  # (2,8)
    B = hcl.placeholder(out_shape, "B", dtype=test_dtype)  # (1,4)
    s = hcl.create_schedule([A], mean_pool)
    f = hcl.build(s)

    np_a = np.random.randint(0, 10, size=in_shape)  # random through 0-10
    print(np_a)

    a = hcl.asarray(np_a, dtype=test_dtype)  # input array np_a
    b = hcl.asarray(np.zeros(out_shape), dtype=test_dtype)  # output array
    f(a, b)
    np_b = b.asnumpy()  # turning output array into array
    b_golden = np.zeros(out_shape)  #
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            b_golden[i, j] = np.mean(
                np_a[i * stride : i * stride + kernel, j * stride : j * stride + kernel]
            )
    assert np.allclose(np_b, b_golden)


def test_meanpool():
    in_shape = (2, 8)
    out_shape = (1, 4)
    stride = 2
    kernel = 2

    test_dtypes = [hcl.Float(32)]

    for test_dtype in test_dtypes:
        _test_meanpool(in_shape, out_shape, stride, kernel, test_dtype)
