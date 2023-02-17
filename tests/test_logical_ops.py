# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np


def test_while_with_and():
    hcl.init()

    def kernel():
        a = hcl.scalar(0, "a", dtype=hcl.UInt(8))
        b = hcl.scalar(0, "a", dtype=hcl.UInt(8))
        res = hcl.scalar(0, "res", dtype=hcl.UInt(8))
        with hcl.while_(hcl.and_(a.v == 0, b.v == 0)):
            res.v = a.v + b.v + 1
            a.v += 1

        with hcl.while_(hcl.and_(a.v == 1, b.v == 0) != 0):
            res.v += 2
            a.v += 1

        return res

    s = hcl.create_schedule([], kernel)
    f = hcl.build(s)
    hcl_res = hcl.asarray(np.zeros((1,), dtype=np.int32), dtype=hcl.UInt(8))
    f(hcl_res)
    assert hcl_res.asnumpy()[0] == 3


def test_if_LogicalOr():
    hcl.init()

    def kernel():
        a = hcl.scalar(0, "a", dtype=hcl.UInt(8))
        b = hcl.scalar(0, "b", dtype=hcl.UInt(8))
        res0 = hcl.scalar(0, "res0", dtype=hcl.UInt(32))
        res1 = hcl.scalar(0, "res1", dtype=hcl.UInt(32))
        with hcl.if_(hcl.or_(a.v == 0, b.v == 0)):  # true
            res0.v = 1
        with hcl.if_(hcl.or_(a.v == 0, b.v == 0) != 0):  # true
            res1.v = 2
        return res0, res1

    s = hcl.create_schedule([], kernel)
    hcl_res0 = hcl.asarray(np.zeros((1,), dtype=np.int32), dtype=hcl.UInt(32))
    hcl_res1 = hcl.asarray(np.zeros((1,), dtype=np.int32), dtype=hcl.UInt(32))
    f = hcl.build(s)
    f(hcl_res0, hcl_res1)
    assert hcl_res0.asnumpy()[0] == 1
    assert hcl_res1.asnumpy()[0] == 2
