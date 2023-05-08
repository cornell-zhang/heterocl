# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np
import pytest


def test_compute():
    shape = (1, 10)

    def loop_body(data, power):
        E = hcl.compute(
            shape, lambda x, y: hcl.power(data[x, y], power[x, y]), "loop_body"
        )
        print(E.loc)
        assert np.equal(str(E.loc), "operation.py:472")

        return E

    A = hcl.placeholder(shape, "A")
    B = hcl.placeholder(shape, "B")
    s = hcl.create_schedule([A, B], loop_body)
    f = hcl.build(s)

    assert np.equal(str(A.loc), "test_operation_loc.py:21") and np.equal(
        str(B.loc), "test_operation_loc.py:22"
    )


def test_scalar():
    def loop_body():
        A = hcl.scalar(0, "A", dtype="uint8")
        assert np.equal(str(A.loc), "operation.py:62")
        return A

    s = hcl.create_schedule([], loop_body)
    f = hcl.build(s)

def test_reduce_axis():
    def loop_body():
        A = hcl.reduce_axis(0, 10, "A")
        assert np.equal(str(A.loc), "operation.py:85")

    s = hcl.create_schedule([], loop_body)
    f = hcl.build(s)

def test_cast():
    def loop_body():
        A = hcl.cast(hcl.UInt(1), 1)
        assert np.equal(str(A.loc), "operation.py:98")

    s = hcl.create_schedule([], loop_body)
    f = hcl.build(s)

def test_const_tensor():
    def loop_body():
        np_A = np.random.randint(1)
        A = hcl.const_tensor(np_A)
        assert np.equal(str(A.loc), "operation.py:107")

    s = hcl.create_schedule([], loop_body)
    f = hcl.build(s)

def test_select():
    def operation_loop():
        D = hcl.select(0, 0, 0)
        assert np.equal(str(D.loc),"operation.py:127")
        return D

    def loop_body(data):
        E = hcl.compute(data.shape, lambda x: operation_loop())
        return E

    A = hcl.placeholder((10,))
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)

@pytest.mark.skip(reason="hcl.make_reduce: module heterocl has no attribute make_reduce")
def test_make_reduce():
    def loop_body():
        A = hcl.make_reduce(0, 0)
        print(A.loc)
        assert np.equal(str(A.loc), "operation.py:256")

    s = hcl.create_schedule([], loop_body)
    f = hcl.build(s)

@pytest.mark.skip(reason="hcl.update: loc of update does not track back to operation.py")
def test_update():
    def loop_body(A, B):
        D = hcl.update(B, lambda x: A[x] + 1)
        print(dir(D))
        assert np.equal(str(D.loc),"operation.py:484")

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))
    s = hcl.create_schedule([A, B], loop_body)
    f = hcl.build(s)

def test_mutate():
    def operation_loop():
        A = hcl.mutate((10,), foo)
        print(A.loc)
        assert np.equal(str(A.loc), "operation.py:496")

    def loop_body(A, B):
        def foo(x):
            with hcl.for_(0, 10) as y:
                with hcl.if_(A[x][y] > 5):
                    B[x] += 1

        A = hcl.mutate((10,), foo)


    A = hcl.placeholder((10, 10))
    B = hcl.placeholder((10,))
    s = hcl.create_schedule([A, B], loop_body)
    f = hcl.build(s)

@pytest.mark.skip(reason="hcl.bitcast: loc of bitcast tracks back to hcl.compute loc")
def test_bitcast():
    def loop_body(A):
        B = hcl.bitcast(A, hcl.Float(32))
        print(B.loc)
        assert np.equal(str(B.loc), "operation.py:515")

        return B

    A = hcl.placeholder((10, 10), dtype=hcl.UInt(32), name="A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)