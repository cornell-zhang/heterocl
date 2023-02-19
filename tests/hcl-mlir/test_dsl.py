# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl


def test_dsl():
    A = hcl.placeholder((32, 32), "A", dtype=hcl.Fixed(12, 6))

    def kernel(A):
        with hcl.for_(0, 32, 1, tag="A") as i:
            with hcl.for_(0, 32, 1) as j:
                with hcl.if_(hcl.and_(i > j, j >= i + 1)):  # affine.if
                    A[i, j] += 1
                with hcl.elif_(A[i, j] == 0):  # scf.if
                    A[i, j] = A[i, j] - 2
                with hcl.else_():
                    A[i, j] = A[i, j] * 3
        return A

    s = hcl.create_schedule([A], kernel)
    code = hcl.build(s, target="vhls")
    print(code)


if __name__ == "__main__":
    test_dsl()
