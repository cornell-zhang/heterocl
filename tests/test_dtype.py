# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
from hcl_mlir import DTypeError
import numpy as np
import pytest


def test_type_comparison():
    # float type attributes
    assert hcl.Float(32).fracs == 23
    assert hcl.Float(32).exponent == 8
    assert hcl.Float(32).bits == 32
    assert hcl.Float(64).fracs == 52
    assert hcl.Float(64).exponent == 11
    assert hcl.Float(64).bits == 64
    # type comparision
    list_of_types = [hcl.Float(32), hcl.Float(64)]
    list_of_types += [hcl.Int(i) for i in range(2, 66, 4)]
    list_of_types += [hcl.UInt(i) for i in range(2, 66, 4)]
    list_of_types += [hcl.Fixed(i, i - 2) for i in range(2, 66, 4)]
    list_of_types += [hcl.UFixed(i, i - 2) for i in range(2, 66, 4)]
    for i in range(len(list_of_types)):
        for j in range(len(list_of_types)):
            if i == j:
                assert list_of_types[i] == list_of_types[j]
            else:
                assert list_of_types[i] != list_of_types[j]


def test_dtype_basic_uint():
    def _test_dtype(dtype):
        hcl.init(dtype)
        np_a = np.random.randint(0, 1 << 63, 100)
        hcl_a = hcl.asarray(np_a)
        np_a2 = hcl_a.asnumpy()

        def cast(val):
            sb = 1 << dtype.bits
            val = val % sb
            return val

        vfunc = np.vectorize(cast)
        np_a3 = vfunc(np_a)
        assert np.allclose(np_a2, np_a3)

    for j in range(0, 10):
        for i in range(2, 66, 4):
            _test_dtype(hcl.UInt(i))


def test_dtype_basic_int():
    def _test_dtype(dtype):
        hcl.init(dtype)
        s = 1 << 63
        np_a = np.random.randint(-s + 1, s, 100)
        hcl_a = hcl.asarray(np_a)
        np_a2 = hcl_a.asnumpy()

        def cast(val):
            sb = 1 << dtype.bits
            sb1 = 1 << (dtype.bits - 1)
            val = val % sb
            val = val if val < sb1 else val - sb
            return val

        vfunc = np.vectorize(cast)
        np_a3 = vfunc(np_a)
        assert np.allclose(np_a2, np_a3)

    for j in range(0, 10):
        for i in range(2, 66, 4):
            _test_dtype(hcl.Int(i))


def test_dtype_basic_ufixed():
    def _test_dtype(dtype):
        hcl.init(dtype)
        np_A = np.random.rand(100)
        hcl_A = hcl.asarray(np_A)
        np_A2 = hcl_A.asnumpy()

        def cast(val):
            sf = 1 << dtype.fracs
            sb = 1 << dtype.bits
            val = val * sf
            val = int(val) % sb
            val = float(val) / sf
            return val

        vfunc = np.vectorize(cast)
        np_A3 = vfunc(np_A)
        assert np.allclose(np_A2, np_A3)

    for j in range(0, 10):
        for i in range(2, 66, 4):
            _test_dtype(hcl.UFixed(i, i - 2))


def test_dtype_overflow_ufixed():
    def _test_dtype(dtype):
        hcl.init(dtype)
        np_A = np.random.rand(100) * 10
        hcl_A = hcl.asarray(np_A)
        np_A2 = hcl_A.asnumpy()

        def cast(val):
            sf = 1 << dtype.fracs
            sb = 1 << dtype.bits
            val = val * sf
            val = int(val) % sb
            val = float(val) / sf
            return val

        vfunc = np.vectorize(cast)
        np_A3 = vfunc(np_A)
        assert np.allclose(np_A2, np_A3)

    for j in range(0, 10):
        for i in range(2, 66, 4):
            _test_dtype(hcl.UFixed(i, i - 2))


def test_dtype_basic_fixed():
    def _test_dtype(dtype):
        hcl.init(dtype)
        np_A = np.random.rand(100) - 0.5
        hcl_A = hcl.asarray(np_A)
        np_A2 = hcl_A.asnumpy()

        def cast(val):
            sf = 1 << dtype.fracs
            sb = 1 << dtype.bits
            sb1 = 1 << (dtype.bits - 1)
            val = val * sf
            val = int(val) % sb
            val = val if val < sb1 else val - sb
            val = float(val) / sf
            return val

        vfunc = np.vectorize(cast)
        np_A3 = vfunc(np_A)
        assert np.allclose(np_A2, np_A3)

    for j in range(0, 10):
        for i in range(2, 66, 4):
            _test_dtype(hcl.Fixed(i, i - 2))


def test_dtype_overflow_fixed():
    def _test_dtype(dtype):
        hcl.init(dtype)
        np_A = (np.random.rand(100) - 0.5) * 100
        hcl_A = hcl.asarray(np_A, dtype)
        np_A2 = hcl_A.asnumpy()

        def cast(val):
            sf = 1 << dtype.fracs
            sb = 1 << dtype.bits
            sb1 = 1 << (dtype.bits - 1)
            val = val * sf
            val = int(val) % sb
            val = val if val < sb1 else val - sb
            val = float(val) / sf
            return val

        vfunc = np.vectorize(cast)
        np_A3 = vfunc(np_A)
        assert np.allclose(np_A2, np_A3)

    for j in range(0, 10):
        for i in range(2, 66, 4):
            _test_dtype(hcl.Fixed(i, i - 2))


def test_dtype_basic_float():
    hcl.init(hcl.Float())
    np_A = np.random.rand(100) - 0.5
    hcl_A = hcl.asarray(np_A)
    np_A2 = hcl_A.asnumpy()
    assert np.allclose(np_A, np_A2)


def test_dtype_compute_fixed():
    def _test_dtype(dtype):
        hcl.init(dtype)
        A = hcl.placeholder((100,))
        B = hcl.placeholder((100,))

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda x: A[x] + B[x])
            D = hcl.compute(A.shape, lambda x: A[x] - B[x])
            E = hcl.compute(A.shape, lambda x: A[x] * B[x])
            # division is not recommended
            # F = hcl.compute(A.shape, lambda x: A[x] / B[x])
            # return C, D, E, F
            return C, D, E

        s = hcl.create_schedule([A, B], kernel)
        f = hcl.build(s)

        np_A = np.random.rand(*A.shape) + 0.1
        np_B = np.random.rand(*B.shape) + 0.1
        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B)
        hcl_C = hcl.asarray(np.zeros(A.shape))
        hcl_D = hcl.asarray(np.zeros(A.shape))
        hcl_E = hcl.asarray(np.zeros(A.shape))
        # hcl_F = hcl.asarray(np.zeros(A.shape))
        # f(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E, hcl_F)
        f(hcl_A, hcl_B, hcl_C, hcl_D, hcl_E)

        np_C = hcl.cast_np(hcl_A.asnumpy() + hcl_B.asnumpy(), dtype)
        np_D = hcl.cast_np(hcl_A.asnumpy() - hcl_B.asnumpy(), dtype)
        np_E = hcl.cast_np(hcl_A.asnumpy() * hcl_B.asnumpy(), dtype)
        # np_F = hcl.cast_np(hcl_A.asnumpy() / hcl_B.asnumpy(), dtype)

        assert np.allclose(np_C, hcl_C.asnumpy())
        assert np.allclose(np_D, hcl_D.asnumpy())
        assert np.allclose(np_E, hcl_E.asnumpy())
        # assert np.allclose(np_F, hcl_F.asnumpy())

    for j in range(0, 10):
        for i in range(6, 66, 4):
            # To avoid floating point exception during division
            _test_dtype(hcl.UFixed(i, i - 2))
            _test_dtype(hcl.Fixed(i, i - 2))


@pytest.mark.skip(reason="Flaky test")
def test_fixed_division():
    def _test_dtype(dtype):
        hcl.init(dtype)
        A = hcl.placeholder((100,))
        B = hcl.placeholder((100,))

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda x: A[x] / B[x])
            return C

        s = hcl.create_schedule([A, B], kernel)
        f = hcl.build(s)

        np_A = np.random.rand(*A.shape) + 0.1
        np_B = np.random.rand(*B.shape) + 0.1
        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B)
        hcl_C = hcl.asarray(np.zeros(A.shape))
        f(hcl_A, hcl_B, hcl_C)

        np_C = hcl.cast_np(hcl_A.asnumpy() / hcl_B.asnumpy(), dtype)

        assert np.allclose(np_C, hcl_C.asnumpy())

    for j in range(0, 10):
        for i in range(6, 32, 4):
            # To avoid floating point exception during division
            _test_dtype(hcl.UFixed(i, i - 2))
            _test_dtype(hcl.Fixed(i, i - 2))


def test_fixed_compute_basic():
    dtype = hcl.Fixed(32, 2)

    hcl.init(dtype)
    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    def kernel(A, B):
        C = hcl.compute(A.shape, lambda x: A[x] + B[x])
        return C

    s = hcl.create_schedule([A, B], kernel)

    f = hcl.build(s)

    np_A = np.random.rand(*A.shape) + 0.1
    np_B = np.random.rand(*B.shape) + 0.1
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_C = hcl.asarray(np.zeros(A.shape))
    f(hcl_A, hcl_B, hcl_C)

    assert np.allclose(hcl_A.asnumpy() + hcl_B.asnumpy(), hcl_C.asnumpy())


def test_dtype_cast():
    def _test_body(dtype1, dtype2, dtype3):
        hcl.init()
        A = hcl.placeholder((2,), dtype=dtype1)
        B = hcl.placeholder((2,), dtype=dtype2)

        def kernel(A, B):
            C = hcl.compute((2,), lambda x: A[x] + B[x], dtype=dtype3)
            D = hcl.compute((2,), lambda x: A[x] - B[x], dtype=dtype3)
            return C, D

        s = hcl.create_schedule([A, B], kernel)
        f = hcl.build(s)

        npA = np.random.rand(2) * 100
        npB = np.random.rand(2) * 100
        npC = np.random.rand(2)
        npD = np.random.rand(2)

        hclA = hcl.asarray(npA, dtype1)
        hclB = hcl.asarray(npB, dtype2)
        hclC = hcl.asarray(npC, dtype3)
        hclD = hcl.asarray(npD, dtype3)

        f(hclA, hclB, hclC, hclD)

        # TODO: check results using HLS CSIM

    from itertools import permutations

    perm = permutations(
        [
            hcl.UInt(1),
            hcl.Int(1),
            hcl.UInt(10),
            hcl.Int(10),
            hcl.UInt(32),
            hcl.Int(32),
            hcl.UFixed(4, 2),
            hcl.Fixed(4, 2),
            hcl.UFixed(32, 16),
            hcl.Fixed(32, 16),
            hcl.Float(),
        ],
        3,
    )

    for dtypes in list(perm):
        _test_body(*dtypes)


def test_dtype_long_int():
    # the longest we can support right now is 2047-bit

    def test_kernel(bw, sl):
        hcl.init(hcl.UInt(32))
        A = hcl.placeholder((100,))
        B = hcl.placeholder((100,))

        def kernel(A, B):
            C = hcl.compute(
                A.shape,
                lambda x: hcl.cast(hcl.UInt(bw), A[x]) << sl,
                dtype=hcl.UInt(bw),
            )
            D = hcl.compute(A.shape, lambda x: B[x] + C[x], dtype=hcl.UInt(bw))
            E = hcl.compute(A.shape, lambda x: A[x])
            return E

        s = hcl.create_schedule([A, B], kernel)
        f = hcl.build(s)
        np_A = np.random.randint(0, 1 << 31, 100)
        np_B = np.random.randint(0, 1 << 31, 100)
        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B)
        hcl_E = hcl.asarray(np.zeros(A.shape))
        f(hcl_A, hcl_B, hcl_E)

        assert np.allclose(np_A, hcl_E.asnumpy())

    test_kernel(64, 30)
    test_kernel(100, 60)
    test_kernel(250, 200)
    test_kernel(500, 400)
    test_kernel(1000, 750)
    test_kernel(2000, 1800)


def test_dtype_too_long_int():
    # the longest we can support right now is 2047-bit

    def test_kernel_total():
        A = hcl.placeholder((100,), dtype=hcl.Int(2048))

    def test_kernel_fracs():
        A = hcl.placeholder((100,), dtype=hcl.Fixed(1000, 800))

    for func in [test_kernel_total, test_kernel_fracs]:
        with pytest.raises(Exception):
            func()


def test_dtype_struct():
    hcl.init()
    A = hcl.placeholder((100,), dtype=hcl.Int(8))
    B = hcl.placeholder((100,), dtype=hcl.Fixed(13, 11))
    C = hcl.placeholder((100,), dtype=hcl.Float())

    def kernel(A, B, C):
        stype = hcl.Struct(
            {"fa": hcl.Int(8), "fb": hcl.Fixed(13, 11), "fc": hcl.Float()}
        )
        D = hcl.compute(A.shape, lambda x: (A[x], B[x], C[x]), dtype=stype)
        E = hcl.compute(A.shape, lambda x: D[x].fa, dtype=hcl.Int(8))
        F = hcl.compute(A.shape, lambda x: D[x].fb, dtype=hcl.Fixed(13, 11))
        G = hcl.compute(A.shape, lambda x: D[x].fc, dtype=hcl.Float())
        # Check the data type
        assert D[0].fa.dtype == hcl.Int(8)
        assert D[0].fb.dtype == hcl.Fixed(13, 11)
        assert D[0].fc.dtype == hcl.Float()
        return E, F, G

    s = hcl.create_schedule([A, B, C], kernel)
    f = hcl.build(s)
    np_A = np.random.randint(0, 500, size=100) - 250
    np_B = np.random.rand(100) - 0.5
    np_C = np.random.rand(100) - 0.5
    np_E = np.zeros(100)
    np_F = np.zeros(100)
    np_G = np.zeros(100)
    hcl_A = hcl.asarray(np_A, dtype=hcl.Int(8))
    hcl_B = hcl.asarray(np_B, dtype=hcl.Fixed(13, 11))
    hcl_C = hcl.asarray(np_C, dtype=hcl.Float())
    hcl_E = hcl.asarray(np_E, dtype=hcl.Int(8))
    hcl_F = hcl.asarray(np_F, dtype=hcl.Fixed(13, 11))
    hcl_G = hcl.asarray(np_G, dtype=hcl.Float())
    f(hcl_A, hcl_B, hcl_C, hcl_E, hcl_F, hcl_G)

    assert np.allclose(hcl_A.asnumpy(), hcl_E.asnumpy())
    assert np.allclose(hcl_B.asnumpy(), hcl_F.asnumpy())
    assert np.allclose(hcl_C.asnumpy(), hcl_G.asnumpy())


def test_dtype_struct_complex():
    hcl.init()
    A = hcl.placeholder((100,))
    B = hcl.placeholder((100,))
    C = hcl.placeholder((100,))
    O = hcl.placeholder((100, 6))

    def kernel(A, B, C, O):
        dtype_xyz = hcl.Struct({"x": hcl.Int(), "y": hcl.Int(), "z": hcl.Int()})
        dtype_out = hcl.Struct(
            {
                "v0": hcl.Int(),
                "v1": hcl.Int(),
                "v2": hcl.Int(),
                "v3": hcl.Int(),
                "v4": hcl.Int(),
                "v5": hcl.Int(),
            }
        )

        D = hcl.compute(A.shape, lambda x: (A[x], B[x], C[x]), dtype=dtype_xyz)
        E = hcl.compute(
            A.shape,
            lambda x: (
                D[x].x * D[x].x,
                D[x].y * D[x].y,
                D[x].z * D[x].z,
                D[x].x * D[x].y,
                D[x].y * D[x].z,
                D[x].x * D[x].z,
            ),
            dtype=dtype_out,
        )
        # with hcl.Stage():
        with hcl.for_(0, 100) as i:
            for j in range(0, 6):
                O[i][j] = E[i].__getattr__("v" + str(j))

    s = hcl.create_schedule([A, B, C, O], kernel)
    f = hcl.build(s)

    np_A = np.random.randint(10, size=100)
    np_B = np.random.randint(10, size=100)
    np_C = np.random.randint(10, size=100)
    np_O = np.zeros((100, 6))

    np_G = np.zeros((100, 6)).astype("int")
    for i in range(0, 100):
        np_G[i][0] = np_A[i] * np_A[i]
        np_G[i][1] = np_B[i] * np_B[i]
        np_G[i][2] = np_C[i] * np_C[i]
        np_G[i][3] = np_A[i] * np_B[i]
        np_G[i][4] = np_B[i] * np_C[i]
        np_G[i][5] = np_A[i] * np_C[i]

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_C = hcl.asarray(np_C)
    hcl_O = hcl.asarray(np_O)
    f(hcl_A, hcl_B, hcl_C, hcl_O)

    assert np.allclose(hcl_O.asnumpy(), np_G)


def test_dtype_bit_slice():
    hcl.init(hcl.Int())

    def kernel():
        A = hcl.compute((10,), lambda x: x)
        assert A[0][0:4].dtype == hcl.UInt(4)
        assert A[0][A[0] : A[4]].dtype == hcl.UInt(4)
        assert A[0][A[0] : A[0] + 4].dtype == hcl.UInt(4)
        return A

    s = hcl.create_schedule([], kernel)
    f = hcl.build(s)
    np_A = np.zeros((10,))
    hcl_A = hcl.asarray(np_A)
    f(hcl_A)


def test_dtype_const_long_int():
    hcl.init(hcl.Int())
    r = np.random.randint(0, 10, size=(1,))

    def kernel():
        A = hcl.compute((1,), lambda x: int(r[0]), dtype=hcl.Int(128))
        B = hcl.compute((1,), lambda x: A[x])
        return B

    s = hcl.create_schedule([], kernel)
    f = hcl.build(s)
    np_B = np.zeros((1,))
    hcl_B = hcl.asarray(np_B)
    f(hcl_B)

    assert np.allclose(r, hcl_B.asnumpy())


def test_dtype_large_array():
    def test_kernel(dtype):
        hcl.init(dtype)

        A = hcl.placeholder((1000,))

        def kernel(A):
            X = hcl.compute(A.shape, lambda x: A[x])
            return hcl.compute(A.shape, lambda x: X[x])

        s = hcl.create_schedule([A], kernel)
        f = hcl.build(s)

        npA = np.random.rand(1000)
        npB = np.zeros(1000)

        hcl_A = hcl.asarray(npA)
        hcl_B = hcl.asarray(npB)

        f(hcl_A, hcl_B)

        assert np.allclose(hcl_A.asnumpy(), hcl_B.asnumpy())

    test_kernel(hcl.Fixed(8, 6))
    test_kernel(hcl.Fixed(16, 14))
    test_kernel(hcl.Fixed(3, 1))
    test_kernel(hcl.Fixed(6, 4))
    test_kernel(hcl.Fixed(11, 9))
    test_kernel(hcl.Fixed(18, 16))
    test_kernel(hcl.Fixed(37, 35))


def test_sign():
    hcl.init(hcl.Int(32))
    A = hcl.placeholder((1, 6, 3, 3))

    def sign(data, name="sign"):
        batch, channel, out_height, out_width = data.shape
        res = hcl.compute(
            (batch, channel, out_height, out_width),
            lambda nn, cc, hh, ww: hcl.select(data[nn, cc, hh, ww] > 0, 1, 0),
            name=name,
            dtype=hcl.UInt(2),
        )
        return res

    s = hcl.create_schedule([A], sign)
    f = hcl.build(s)

    np_A = np.random.randint(0, 4, size=(1, 6, 3, 3))

    hcl_A = hcl.asarray(np_A, dtype=hcl.Int(32))
    hcl_B = hcl.asarray(np.zeros((1, 6, 3, 3), dtype="int"), dtype=hcl.UInt(2))

    f(hcl_A, hcl_B)


def test_struct_scalar():
    hcl.init()

    def kernel():
        stype = hcl.Struct({"x": hcl.UInt(8), "y": hcl.UInt(8)})
        xy = hcl.scalar(0x1234, "foo", dtype=stype).v
        z1 = hcl.compute((2,), lambda i: 0, dtype=hcl.UInt(32))
        z1[0] = xy.y
        z1[1] = xy.x
        assert xy.y.dtype == hcl.UInt(8)
        assert xy.x.dtype == hcl.UInt(8)
        return z1

    s = hcl.create_schedule([], kernel)
    hcl_res = hcl.asarray(np.zeros((2,), dtype=np.uint32), dtype=hcl.UInt(32))
    f = hcl.build(s)
    f(hcl_res)
    np_res = hcl_res.asnumpy()
    golden = np.zeros((2,), dtype=np.uint32)
    golden[0] = 0x12
    golden[1] = 0x34
    assert np.array_equal(golden, np_res)


def test_struct_uint():
    hcl.init()
    rshape = (1,)

    def kernel():
        stype = hcl.Struct({"x": "uint8", "y": "uint8"})
        ival = hcl.cast("uint16", 0)
        d = hcl.scalar(ival, "d", dtype=stype).v
        r = hcl.compute(rshape, lambda _: 0, dtype=hcl.UInt(32))
        stype_nested = hcl.Struct({"x": "uint8", "y": stype})
        d_nested = hcl.scalar(hcl.cast("uint24", ival), "d_nested", dtype=stype_nested)
        r[0] = d.x
        return r

    s = hcl.create_schedule([], kernel)
    hcl_res = hcl.asarray(np.zeros(rshape, dtype=np.uint32), dtype=hcl.UInt(32))
    f = hcl.build(s)
    f(hcl_res)
    np_res = hcl_res.asnumpy()
    golden = np.zeros(rshape, dtype=np.uint32)
    golden[0] = 0
    assert np.array_equal(golden, np_res)


def test_bitand_type():
    hcl.init()

    def kernel():
        x = hcl.scalar(0, "x", dtype="uint16")

        def do(i):
            with hcl.if_((x.v & 0x15) == 1):  # test regular variable
                hcl.print((), "A\n")
            with hcl.if_((i & 1) == 1):  # test index
                hcl.print((), "B\n")

        hcl.mutate((1,), do)
        r = hcl.compute((2,), lambda i: 0, dtype=hcl.UInt(32))
        return r

    s = hcl.create_schedule([], kernel)
    hcl_res = hcl.asarray(np.zeros((2,), dtype=np.uint32), dtype=hcl.UInt(32))
    f = hcl.build(s)
    f(hcl_res)


def test_int_to_fixed_cast():
    for in_precision in range(6, 33, 2):
        for out_precision in range(8, 33, 2):
            in_dtype, out_dtype = hcl.Int(in_precision), hcl.Fixed(
                out_precision, int(out_precision / 2)
            )

            def cast(A):
                casted = hcl.compute(
                    A.shape,
                    lambda *args: hcl.cast(out_dtype, A[args]),
                    "cast",
                    dtype=out_dtype,
                )
                return casted

            A = hcl.placeholder((10,), "A", dtype=in_dtype)
            s = hcl.create_schedule([A], cast)
            f = hcl.build(s)

            A_np = np.random.randint(0, 5, A.shape)

            A_hcl = hcl.asarray(A_np, dtype=in_dtype)
            C_hcl = hcl.asarray(np.zeros(A.shape), dtype=out_dtype)

            f(A_hcl, C_hcl)

            A_np = A_hcl.asnumpy()
            C_np = C_hcl.asnumpy()

            if not np.allclose(A_np, C_np):
                print(
                    "{} -> {} failed, wrong result value".format(in_dtype, out_dtype),
                    flush=True,
                )
                print("A_np: ", A_np)
                print("C_np: ", C_np)
                assert False, "test failed, see failed test case above"


def test_expected_fp_as_index_error():
    def fp_as_slice_idx(A):
        a = hcl.scalar(0)
        b = hcl.scalar(2)
        idx = hcl.power(b.v, a.v)
        # idx is float, should raise DTypeError when used as index
        B = hcl.compute((1,), lambda _: A[idx])
        return B

    def fp_as_bit_idx(A):
        start = hcl.scalar(0, dtype=hcl.Float())
        end = hcl.scalar(2, dtype=hcl.Float())
        B = hcl.compute((1,), lambda _: A[0][start.v : end.v])
        return B

    with pytest.raises(DTypeError):
        A = hcl.placeholder((2,), "A")
        hcl.customize([A], fp_as_slice_idx)

    with pytest.raises(DTypeError):
        A = hcl.placeholder((1,), "A", dtype=hcl.UInt(32))
        hcl.customize([A], fp_as_bit_idx)


def test_fp_to_index_cast():
    def kernel(A):
        a = hcl.scalar(0)
        b = hcl.scalar(2)
        idx = hcl.power(b.v, a.v)
        idx = hcl.cast(hcl.Index(), idx)
        B = hcl.compute((1,), lambda _: A[idx])
        return B

    A = hcl.placeholder((2,), "A")
    s = hcl.create_schedule([A], kernel)
    f = hcl.build(s)
    np_A = hcl.asarray([0b10101100, 0b01100101])
    np_B = hcl.asarray([0])
    f(np_A, np_B)
    assert np_B.asnumpy().tolist() == [0b01100101]


if __name__ == "__main__":
    pytest.main([__file__])
