# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np
import pytest


def test_placeholder():
    shape = (1, 10)
    A = hcl.placeholder(shape, "A")
    B = hcl.placeholder(shape, "B")

    assert np.equal(str(A.loc), "test_ast_loc.py:7") and np.equal(
        str(B.loc), "test_ast_loc.py:8"
    )


# Testing intrin.py


def test_exp():
    shape = (1, 10)

    def math_loop(x):
        D = hcl.exp(x)
        assert np.equal(str(D.loc), "intrin.py:11")
        return D

    def loop_body(data):
        E = hcl.compute(shape, lambda x, y: math_loop(data[x, y]), "loop_body")
        return E

    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)


def test_power():
    shape = (1, 10)

    def math_loop(x, y):
        D = hcl.power(x, y)
        assert np.equal(str(D.loc), "intrin.py:17")
        return D

    def loop_body(data, power):
        E = hcl.compute(
            shape, lambda x, y: math_loop(data[x, y], power[x, y]), "loop_body"
        )
        return E

    A = hcl.placeholder(shape, "A")
    B = hcl.placeholder(shape, "B")
    s = hcl.create_schedule([A, B], loop_body)
    f = hcl.build(s)


def test_log():
    shape = (1, 10)

    def math_loop(x):
        D = hcl.log(x)
        assert np.equal(str(D.loc), "intrin.py:23")
        return D

    def loop_body(data):
        E = hcl.compute(shape, lambda x, y: math_loop(data[x, y]), "loop_body")
        return E

    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)


def test_log2():
    shape = (1, 10)

    def math_loop(x):
        D = hcl.log2(x)
        assert np.equal(str(D.loc), "intrin.py:29")
        return D

    def loop_body(data):
        E = hcl.compute(shape, lambda x, y: math_loop(data[x, y]), "loop_body")
        return E

    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)


def test_log10():
    shape = (1, 10)

    def math_loop(x):
        D = hcl.log10(x)
        assert np.equal(str(D.loc), "intrin.py:35")
        return D

    def loop_body(data):
        E = hcl.compute(shape, lambda x, y: math_loop(data[x, y]), "loop_body")
        return E

    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)


def test_sqrt():
    shape = (1, 10)

    def math_loop(x):
        D = hcl.sqrt(x)
        assert np.equal(str(D.loc), "intrin.py:41")
        return D

    def loop_body(data):
        E = hcl.compute(shape, lambda x, y: math_loop(data[x, y]), "loop_body")
        return E

    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)


def test_sin():
    shape = (1, 10)

    def math_loop(x):
        D = hcl.sin(x)
        assert np.equal(str(D.loc), "intrin.py:47")
        return D

    def loop_body(data):
        E = hcl.compute(shape, lambda x, y: math_loop(data[x, y]), "loop_body")
        return E

    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)


def test_cos():
    shape = (1, 10)

    def math_loop(x):
        D = hcl.cos(x)
        assert np.equal(str(D.loc), "intrin.py:53")
        return D

    def loop_body(data):
        E = hcl.compute(shape, lambda x, y: math_loop(data[x, y]), "loop_body")
        return E

    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)


def test_tanh():
    shape = (1, 10)

    def math_loop(x):
        D = hcl.tanh(x)
        assert np.equal(str(D.loc), "intrin.py:69")
        return D

    def loop_body(data):
        E = hcl.compute(shape, lambda x, y: math_loop(data[x, y]), "loop_body")
        return E

    A = hcl.placeholder(shape, "A")
    s = hcl.create_schedule([A], loop_body)
    f = hcl.build(s)


# Testing expressions in ast.py
