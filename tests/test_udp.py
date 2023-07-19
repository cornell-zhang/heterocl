# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test user-defined primitives"""

import heterocl as hcl
import heterocl.ir.transform as hir


@hcl.register_primitive()
class BufferRootPrimitive(hcl.Primitive):
    name = "buffer_root"

    @staticmethod
    def apply(sch):
        loops = hir.get_affine_loop_nests(sch.top_func)[0]
        for i, arg in enumerate(sch.top_func.arguments):
            hir.create_buffer(arg, f"arg_{i}", ip=loops[0][1])


@hcl.register_primitive()
class AnnotatePrimitive(hcl.Primitive):
    name = "annotate"

    @staticmethod
    def apply(sch):
        loops = hir.get_affine_loop_nests(sch.top_func)
        for i, band in enumerate(loops):
            for j, (name, loop) in enumerate(band):
                hir.annotate(loop, f"Loop_{i}{j}")


def test_gemm_buffer(M=32, N=32, K=32, dtype=hcl.Int(), target=None):
    hcl.init(hcl.Float())
    A = hcl.placeholder((M, K), name="A")
    B = hcl.placeholder((K, N), name="B")

    def gemm(A, B):
        k = hcl.reduce_axis(0, K, name="k")
        C = hcl.compute((M, N), lambda i, j: hcl.sum(A[i, k] * B[k, j], axis=k), "C")
        return C

    s = hcl.create_schedule([A, B], gemm)

    # optimization
    C = gemm.C
    s[C].reorder(C.axis[1], C.axis[0])
    s.buffer_root()
    s.annotate()
    hcl.build(s, target="vhls")
    print(s.module)


if __name__ == "__main__":
    test_gemm_buffer()
