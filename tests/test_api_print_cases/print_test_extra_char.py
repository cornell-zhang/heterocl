# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl


def test_print_extra_char():
    hcl.init()

    def kernel():
        ip = hcl.scalar(10, "ip", "uint32")
        cnt = hcl.scalar(10, "tel", "uint32")
        a1 = hcl.scalar(1, "a1", "uint11")
        a2 = hcl.scalar(265, "y1", "uint21")

        hcl.print((), "                    ce0.mf.UC ")
        hcl.print((ip.v), "ip=%d ")
        hcl.print((cnt.v), "cnt=%d ")
        hcl.print((a1.v, a2.v), "mload spadaddr=%d hbmaddr=%d")
        hcl.print((), "    \n")

    s = hcl.create_schedule([], kernel)
    f = hcl.build(s)
    f()


test_print_extra_char()
