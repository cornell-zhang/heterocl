# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np


def test_nested_if_else():
    hcl.init()

    def kernel():
        x = hcl.scalar(3, "x", dtype="uint8")
        z = hcl.scalar(0, "z", dtype="uint1")

        def rec(y):
            if y == 3:
                z.v = 1
            else:
                with hcl.if_(x.v == y):
                    z.v = 0
                with hcl.else_():
                    rec(y + 1)

        rec(0)
        r = hcl.compute((1,), lambda i: z.v, dtype=hcl.UInt(32))
        return r

    s = hcl.create_schedule([], kernel)
    hcl_res = hcl.asarray(np.zeros((1,), dtype=np.uint32), dtype=hcl.UInt(32))
    f = hcl.build(s)
    f(hcl_res)
    np_res = np.zeros((1,), dtype=np.uint32)
    np_res[0] = 1
    assert np.array_equal(hcl_res.asnumpy(), np_res)


def test_nested_if_if_else():
    hcl.init()

    def kernel():
        a = hcl.scalar(0, "a", dtype="uint8")
        x = hcl.scalar(0, "x", dtype="uint8")
        z = hcl.scalar(0, "z", dtype="uint1")

        def rec(y):
            if y == 3:
                z.v = 1
            else:
                with hcl.if_(x.v == y):
                    z.v = 0
                    with hcl.if_(a.v == 0):  # add this if/else statement
                        a.v = y
                    with hcl.else_():
                        a.v = y + 1
                with hcl.else_():
                    rec(y + 1)

        rec(0)
        r = hcl.compute((2,), lambda i: z.v, dtype=hcl.UInt(32))
        return r

    s = hcl.create_schedule([], kernel)
    hcl_res = hcl.asarray(np.zeros((2,), dtype=np.uint32), dtype=hcl.UInt(32))
    f = hcl.build(s)
    f(hcl_res)


def test_nested_if_mutate():
    hcl.init()

    def kernel():
        a = hcl.scalar(0, "a", dtype="uint8")
        x = hcl.scalar(0, "x", dtype="uint8")
        z = hcl.scalar(0, "z", dtype="uint1")

        def rec(y):
            if y == 3:
                z.v = 1
            else:
                with hcl.if_(x.v == y):

                    def doit(i):
                        z.v = 0
                        with hcl.if_(a.v == 0):
                            a.v = y
                        with hcl.else_():
                            a.v = y + 1

                    hcl.mutate((1,), doit)
                with hcl.else_():
                    rec(y + 1)

        rec(0)
        r = hcl.compute((2,), lambda i: z.v, dtype=hcl.UInt(32))
        return r

    s = hcl.create_schedule([], kernel)
    hcl_res = hcl.asarray(np.zeros((2,), dtype=np.uint32), dtype=hcl.UInt(32))
    f = hcl.build(s)
    f(hcl_res)


def test_nested_if_elif_else():
    hcl.init()

    def kernel(x, y, z):
        with hcl.if_(x.v == 1):
            with hcl.if_(y.v == 1):
                z.v = 11
            with hcl.elif_(y.v == 2):
                z.v = 12
            with hcl.else_():
                z.v = 13
        with hcl.elif_(x.v == 2):
            with hcl.if_(y.v == 1):
                z.v = 21
            with hcl.elif_(y.v == 2):
                z.v = 22
            with hcl.else_():
                z.v = 23
        with hcl.else_():
            with hcl.if_(y.v == 1):
                z.v = 31
            with hcl.elif_(y.v == 2):
                z.v = 32
            with hcl.else_():
                z.v = 33

    x = hcl.placeholder((1,), "x", dtype=hcl.UInt(32))
    y = hcl.placeholder((1,), "y", dtype=hcl.UInt(32))
    z = hcl.placeholder((1,), "z", dtype=hcl.UInt(32))
    s = hcl.create_schedule([x, y, z], kernel)
    f = hcl.build(s)
    for x_v in [1, 2, 3]:
        for y_v in [1, 2, 3]:
            np_x = np.array([x_v], dtype=np.uint32)
            np_y = np.array([y_v], dtype=np.uint32)
            np_z = np.array([0], dtype=np.uint32)
            hcl_x = hcl.asarray(np_x, dtype=hcl.UInt(32))
            hcl_y = hcl.asarray(np_y, dtype=hcl.UInt(32))
            hcl_z = hcl.asarray(np_z, dtype=hcl.UInt(32))
            f(hcl_x, hcl_y, hcl_z)
            golden = x_v * 10 + y_v
            assert hcl_z.asnumpy()[0] == golden
