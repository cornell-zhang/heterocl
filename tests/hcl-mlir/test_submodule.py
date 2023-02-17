# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl

# from submodule import submodule # global import

target = "vhls"

hcl.init()


def main():
    A = hcl.placeholder((10, 10), name="A")
    B = hcl.placeholder((10, 10), name="B")

    def algo(A, B):
        from submodule import submodule  # this local import is required

        submodule(A)
        hcl.update(B, lambda *args: B[args] + A[args])

    s = hcl.create_schedule([A, B], func=algo, name="main")
    f = hcl.build(s, target=target, name="main")
    print(f)


if __name__ == "__main__":
    main()
