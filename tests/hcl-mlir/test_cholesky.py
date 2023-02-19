# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl

hcl.init(hcl.Float())


def top_cholesky(N, dtype=hcl.Int(), target=None):
    hcl.init(dtype)
    A = hcl.placeholder((N, N), "A")

    def kernel_cholesky(A):
        with hcl.for_(0, N, name="i") as i:
            # Case: j < i
            with hcl.for_(0, i, name="j") as j:
                with hcl.for_(0, j, name="k") as k:
                    A[i, j] = A[i, j] - A[i, k] * A[j, k]
                A[i, j] = A[i, j] / A[j, j]
            # Case: i == j
            with hcl.for_(0, i, name="k") as k:
                A[i, i] = A[i, i] - A[i, k] * A[i, k]
            A[i, i] = hcl.sqrt(A[i, i] * 1.0)
        return A

    s = hcl.create_schedule([A], kernel_cholesky)
    return hcl.build(s, target=target)


if __name__ == "__main__":
    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="csyn", project="sobel.prj")
    mod = top_cholesky(1024, dtype=hcl.Float(), target=target)
    print(mod.src)
