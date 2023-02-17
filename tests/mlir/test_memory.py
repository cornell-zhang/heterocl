# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os, psutil
import heterocl as hcl
import numpy as np


def test_memory_consumption():
    hcl.init()

    def compute_in_loop(bound):
        def kernel():
            r = hcl.compute((1,), lambda i: 0, dtype=hcl.UInt(32))
            cnt = hcl.scalar(0, "cnt", dtype="uint32")
            with hcl.while_(cnt.v < bound):
                cnt.v = cnt.v + 1
                tmp = hcl.compute((8192,), lambda i: 0, "tmp", dtype="uint32")
                r[0] = r[0] + tmp[0]
            return r

        s = hcl.create_schedule([], kernel)
        f = hcl.build(s)
        hcl_res = hcl.asarray(np.zeros((1,), dtype=np.uint32), dtype=hcl.UInt(32))
        f(hcl_res)
        process = psutil.Process(os.getpid())
        mb = process.memory_info().rss // (1024 * 1024)
        return mb

    mb1 = compute_in_loop(100)
    mb2 = compute_in_loop(100000)
    assert mb2 < 2 * mb1
