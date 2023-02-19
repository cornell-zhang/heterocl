# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np


# Related to issue #148
def test_single_load_cond_if():
    hcl.init()
    rshape = (1,)

    def kernel():
        inst_id = hcl.scalar(0, "inst_id", dtype=hcl.UInt(16)).v
        add_expr = hcl.scalar(1, "id", dtype=hcl.UInt(16)).v + 1
        mul_expr = add_expr * 2
        r = hcl.compute(rshape, lambda _: 0, dtype=hcl.UInt(32))
        with hcl.if_(inst_id == 0):
            r[0] = 1
        with hcl.if_(inst_id == 1):
            r[0] = 2
        with hcl.if_(add_expr == 1):
            r[0] = 3
        with hcl.if_(mul_expr == 0):
            r[0] = 4
        return r

    s = hcl.create_schedule([], kernel)
    hcl_res = hcl.asarray(np.zeros(rshape, dtype=np.uint32), dtype=hcl.UInt(32))
    f = hcl.build(s)
    f(hcl_res)
    np_res = np.zeros(rshape, dtype=np.uint32)
    np_res[0] = 1
    assert np.array_equal(hcl_res.asnumpy(), np_res)


def test_single_load_cond_elif():
    hcl.init()
    rshape = (1,)

    def kernel():
        inst_id = hcl.scalar(2, "inst_id", dtype=hcl.UInt(16)).v
        r = hcl.compute(rshape, lambda _: 0, dtype=hcl.UInt(32))
        with hcl.if_(inst_id == 0):
            r[0] = 1
        with hcl.elif_(inst_id == 1):
            r[0] = 2
        with hcl.elif_(inst_id == 2):
            r[0] = 3
        with hcl.else_():
            r[0] = 4
        return r

    s = hcl.create_schedule([], kernel)
    hcl_res = hcl.asarray(np.zeros(rshape, dtype=np.uint32), dtype=hcl.UInt(32))
    f = hcl.build(s)
    f(hcl_res)
    np_res = np.zeros(rshape, dtype=np.uint32)
    np_res[0] = 3
    assert np.array_equal(hcl_res.asnumpy(), np_res)


def test_mixed_load_cond_elif():
    hcl.init()
    rshape = (1,)

    def kernel():
        inst_id = hcl.scalar(2, "inst_id", dtype=hcl.UInt(16))
        inst_id_v = inst_id.v
        r = hcl.compute(rshape, lambda _: 0, dtype=hcl.UInt(32))
        with hcl.if_(inst_id_v == 0):
            r[0] = 1
        with hcl.if_(inst_id[0] == 1):
            r[0] = 2
        with hcl.elif_(inst_id_v == 2):
            r[0] = 3
        with hcl.else_():
            r[0] = 4
        return r

    s = hcl.create_schedule([], kernel)
    hcl_res = hcl.asarray(np.zeros(rshape, dtype=np.uint32), dtype=hcl.UInt(32))
    f = hcl.build(s)
    f(hcl_res)
    np_res = np.zeros(rshape, dtype=np.uint32)
    np_res[0] = 3
    assert np.array_equal(hcl_res.asnumpy(), np_res)


# Related to issue #141
def test_if_struct_access():
    hcl.init()

    def kernel():
        # tag = hcl.scalar(0, "tag", dtype='uint32')
        res = hcl.compute((1,), lambda _: 0, name="res", dtype="uint32")
        stype = hcl.Struct({"x": hcl.UInt(8), "y": hcl.UInt(8)})
        s_xy = hcl.scalar(0x1234, "xy", dtype=stype)
        xy = s_xy.v
        x = xy.x
        y = xy.y
        # use x and y
        with hcl.if_(x == 1):
            res[0] = 1
        with hcl.elif_(y == 2):
            res[0] = 2
        # use xy.x and xy.y
        with hcl.elif_(xy.x == 3):
            res[0] = 3
        with hcl.elif_(xy.y == 4):
            res[0] = 4
        # use s_xy.v.x and s_xy.v.y
        with hcl.elif_(s_xy.v.x == 5):
            res[0] = 5
        with hcl.elif_(s_xy.v.y == 6):
            res[0] = 6
        with hcl.else_():
            res[0] = 7
        # use get bit and get slice
        with hcl.if_(x[0] == 1):
            res[0] = 8
        with hcl.elif_(y[1] == 1):
            res[0] = 9
        with hcl.elif_(xy.x[0:2].reverse() == 1):
            res[0] = 10
        return res

    s = hcl.create_schedule([], kernel)
    hcl.lower(s)


# Related to issue #155
def test_print_before_if():
    hcl.init()
    rshape = (1,)

    def kernel():
        stype = hcl.Struct({"x": hcl.UInt(8), "y": hcl.UInt(8)})
        xy = hcl.scalar(0x1234, "foo", dtype=stype).v
        hcl.print((xy.x, xy.y), "match 0: %d %d\n")
        with hcl.if_(xy.x == 0):
            pass
        r = hcl.compute(rshape, lambda _: 0, dtype=hcl.Int(32))
        return r

    s = hcl.create_schedule([], kernel)
    hcl_res = hcl.asarray(np.zeros(rshape, dtype=np.uint32), dtype=hcl.UInt(32))
    f = hcl.build(s)


def test_fresh_load():
    def kernel(a, x):
        def outer(i):
            const = hcl.scalar(64, "const").v

            with hcl.for_(0, 10) as j:
                v1 = hcl.scalar(i * const + 1, "v1")

            with hcl.for_(0, 10) as j:
                v2 = hcl.scalar(i * const + 2, "v2")

        hcl.mutate((10,), outer, "outer")

    a = hcl.placeholder((1,), "a")
    x = hcl.placeholder((1,), "x")
    s = hcl.create_schedule([a, x], kernel)
    f = hcl.build(s)
