# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
from heterocl.platforms import import_json_platform
import hcl_mlir
import numpy as np
import os
import pytest

dir_path = os.path.dirname(os.path.realpath(__file__))


def test_print_platform_hierarchy():
    target = import_json_platform(
        os.path.join(dir_path, "test_platform_spec/xilinx_u280.json")
    )
    target_mode = "csyn"
    hcl.init()
    A = hcl.placeholder((10, 32), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda *args: A[args] + 1, "B")
        C = hcl.compute(A.shape, lambda *args: B[args] + 1, "C")
        return C

    s = hcl.create_schedule([A], kernel)
    s.to(A, target.xcel)
    s.to(kernel.C, target.host)

    target.config(compiler="vivado_hls", mode="csyn", project="hlscode.prj")
    f = hcl.build(s, target)

    if os.system("which vivado_hls >> /dev/null") != 0:
        return

    np_A = np.random.randint(10, size=(10, 32))
    np_B = np.zeros((10, 32))

    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))
    f(hcl_A, hcl_B)
    ret_B = hcl_B.asnumpy()

    if "csyn" in target_mode:
        report = f.report()
        assert "ReportVersion" in report

    elif "csim" in target_mode:
        np.testing.assert_array_equal(ret_B, (np_A + 2) * 2)


if __name__ == "__main__":
    test_print_platform_hierarchy()
