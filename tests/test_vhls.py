import os
import sys
import json
import heterocl as hcl

def test_basic():
    A = hcl.placeholder((1, 32, 14, 14), dtype=hcl.UInt(1), name="A")
    B = hcl.placeholder((64, 32, 3, 3), dtype=hcl.UInt(1), name="B")
    rc = hcl.reduce_axis(0, 32)
    ry = hcl.reduce_axis(0, 3)
    rx = hcl.reduce_axis(0, 3)
    C = hcl.compute((1, 64, 12, 12),
        lambda nn, ff, yy, xx: hcl.sum(
            A[nn, rc, yy + ry, xx + rx] * B[ff, rc, ry, rx], axis=[rc, ry, rx]),
        dtype=hcl.UInt(8), name="C")
    s = hcl.create_schedule([A, B, C])
    code = hcl.build(s, target='vhls')
    print code


if __name__ == '__main__':
    test_basic()
