# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl

hcl.init(hcl.Fixed(12, 6))


def test_fixed():
    A = hcl.placeholder((32, 32), "A")

    def kernel(A):
        with hcl.for_(0, 32, 1, "i") as i:
            with hcl.for_(0, 32, 1, "j") as j:
                A[i, j] = A[i, j] + 1
        return A

    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="csyn", project="fixed.prj")

    s = hcl.create_schedule([A], kernel)
    mod = hcl.build(s, target=target)
    print(mod.src)


if __name__ == "__main__":
    test_fixed()
