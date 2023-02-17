# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np
import pytest

hcl.init(hcl.Float(32))


def test_vadd(target=None):
    n = 2
    A = hcl.placeholder((n, n), "A")
    B = hcl.placeholder((n, n), "B")

    def vadd(A, B):
        C = hcl.compute((n, n), lambda *i: A[i] + B[i], "C")
        return C

    s = hcl.create_schedule([A, B], vadd)
    f = hcl.build(s, target)
    m1 = hcl.asarray(
        np.random.randint(10, size=(n, n)).astype(np.float32), hcl.Float(32)
    )
    m2 = hcl.asarray(
        np.random.randint(10, size=(n, n)).astype(np.float32), hcl.Float(32)
    )
    m3 = hcl.asarray(np.zeros((n, n)).astype(np.float32), hcl.Float(32))
    f(m1, m2, m3)
    golden = m1.asnumpy() + m2.asnumpy()
    assert np.allclose(golden, m3.asnumpy()), "test_vadd failed."


def test_vsum(target=None):
    n = 2
    A = hcl.placeholder((n,), "A")

    def sum(A):
        x = hcl.reduce_axis(0, n, "x")
        return hcl.compute((1,), lambda *_: hcl.sum(A[x], axis=x), "sum")

    s = hcl.create_schedule([A], sum)
    f = hcl.build(s, target)
    m1 = hcl.asarray(np.random.randint(10, size=(n,)).astype(np.float32), hcl.Float(32))
    m2 = hcl.asarray(np.zeros((1,)).astype(np.float32), hcl.Float(32))
    f(m1, m2)
    golden = np.sum(m1.asnumpy())
    assert np.isclose(golden, m2.asnumpy()[0]), "test_vsum failed."


if __name__ == "__main__":
    test_vadd()
    test_vsum()
