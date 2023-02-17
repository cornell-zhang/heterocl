# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np


def test_name_conflict():
    hcl.init()

    def funcA(z):
        tmp = hcl.scalar(0, "tmp", dtype="uint16")
        return z + tmp.v

    def kernel():
        x = funcA(0)
        y = funcA(1)
        r = hcl.compute((2,), lambda i: 0, dtype=hcl.UInt(32))
        return r

    s = hcl.create_schedule([], kernel)
    ir_str = str(hcl.lower(s))
    assert "tmp_0" in ir_str
    hcl_res = hcl.asarray(np.zeros((2,), dtype=np.uint32), dtype=hcl.UInt(32))
    f = hcl.build(s)
    f(hcl_res)
