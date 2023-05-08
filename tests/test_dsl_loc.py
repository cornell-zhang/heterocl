# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np
import pytest

def test_and():
    shape = (1,10)
    def dsl_loop(x,y):
        D = hcl.and_(x,y)
        assert np.equal(str(D.loc),"dsl.py:30")
        return D

    def loop_body(data, power):
        E = hcl.compute(shape, lambda x,y: dsl_loop(data[x,y],power[x,y]), "loop_body")
        return E

    A = hcl.placeholder(shape, "A")
    B = hcl.placeholder(shape, "B")
    s = hcl.create_schedule([A, B], loop_body)
    f = hcl.build(s)

def test_or():
    shape = (1,10)
    def dsl_loop(x,y):
        D = hcl.or_(x,y)
        assert np.equal(str(D.loc),"dsl.py:42")
        return D

    def loop_body(data, power):
        E = hcl.compute(shape, lambda x,y: dsl_loop(data[x,y],power[x,y]), "loop_body")
        return E

    A = hcl.placeholder(shape, "A")
    B = hcl.placeholder(shape, "B")
    s = hcl.create_schedule([A, B], loop_body)
    f = hcl.build(s)

def test_not():
    shape = (1,1)
    def dsl_loop(x):
        D = hcl.not_(x)
        assert np.equal(str(D.loc),"dsl.py:54")
        return D

    def loop_body(data):
        E = hcl.compute(shape, lambda x,y: dsl_loop(data[x,y]), "loop_body")
        return E

    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)

@pytest.mark.skip(reason="hcl.for_: no loc attribute")
def test_for():
    shape = (1, 10)

    def dsl_loop(x, y):
        D = hcl.for_(x, y)
        print(dir(D))
        print(D.__ne__)
        assert np.equal(str(D), "dsl.py:70")
        return D

    def loop_body(x):
        with dsl_loop(0, 10) as j:
            E = hcl.add(j + x)

    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)


@pytest.mark.skip(reason="hcl.if_: no loc attribute")
def test_if():

    def dsl_loop(A):
        D = hcl.if_(A[0] > 5)
        print(dir(D))
        assert np.equal(str(D), "dsl.py:84")
        return D

    def loop_body(A):
        with dsl_loop(A):
            A[0] = 5

    A = hcl.placeholder((1,))
    s = hcl.create_schedule(A, loop_body)
    f = hcl.build(s)


@pytest.mark.skip(reason="hcl.else_: no loc attribute")
def test_else():
    shape = (1, 10)

    def dsl_loop():
        D = hcl.else_()
        print(dir(D))
        assert np.equal(str(D), "dsl.py:97")
        return D

    def loop_body(A):
        with hcl.if_(A[0] > 5):
            A[0] = 5
        with dsl_loop():
            A[0] = -1

    A = hcl.placeholder((1,))
    s = hcl.create_schedule(A, loop_body)
    f = hcl.build(s)

@pytest.mark.skip(reason="hcl.elif_: no loc attribute")
def test_elif():
    def dsl_loop(A):
        D = hcl.elif_(A[0] > 3)
        print(dir(D))
        assert np.equal(str(D.loc),"dsl.py:110")
        return D

    def loop_body(A):
        with hcl.if_(A[0] > 5):
            A[0] = 5
        with dsl_loop(A):
            A[0] = 3

    A = hcl.placeholder((1,))
    s = hcl.create_schedule(A, loop_body)
    f = hcl.build(s)

@pytest.mark.skip(reason="hcl.while_: no loc attribute")
def test_while():
    shape = (1,10)
    def dsl_loop(a):
        D = hcl.while_(a[0] < 10)
        print(dir(D))
        assert np.equal(str(D.loc),"dsl.py:123")
        return D

    def loop_body(A):
        a = hcl.scalar(0)
        with dsl_loop(a):
            A[a[0]] = a[0]
            a[0] += 1

    A = hcl.placeholder((10,))
    s = hcl.create_schedule(A, loop_body)
    f = hcl.build(s)

@pytest.mark.skip(reason="hcl.def_: no loc attribute")
def test_def():
    shape = (1,10)
    def dsl_loop():
        D = hcl.def_()
        print(dir(D))
        assert np.equal(str(D.loc),"dsl.py:139")
        return D
    
    def loop_body(A, B):
        dsl_loop()
        def update_B(A, B, x):
            B[x] = A[x] + 1

        with hcl.for_(0, 10) as i:
            update_B(A, B, i)

    A = hcl.placeholder((10,))
    B = hcl.placeholder((10,))

    s = hcl.create_schedule([A, B], loop_body)
    f = hcl.build(s)

@pytest.mark.skip(reason="hcl.return_: no loc attribute")
def test_return():
    def dsl_loop(A):
        D = hcl.return_(5)
        print(dir(D))
        assert np.equal(str(D), "dsl.py:201")

    def loop_body(A):
        with hcl.if_(A[0] > 5):
            dsl_loop(A)
            

    A = hcl.placeholder((1,))
    s = hcl.create_schedule(A, loop_body)
    f = hcl.build(s)