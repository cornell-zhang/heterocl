# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl


def test_print_tensor_element():
    hcl.init()

    def kernel():
        z1 = hcl.compute((2, 3, 4), lambda x, y, z: 0, dtype=hcl.Int(32))

        def do(i, j, k):
            z1[0, 0, 0] = 53

        hcl.mutate(z1.shape, do)
        hcl.print((z1[0, 0, 0]), "here %d\n")

    s = hcl.create_schedule([], kernel)
    f = hcl.build(s)
    f()


test_print_tensor_element()
