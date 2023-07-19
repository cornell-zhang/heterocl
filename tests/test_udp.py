# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test user-defined primitives"""

import heterocl as hcl
import hcl_mlir
from hcl_mlir.ir import *
from hcl_mlir.dialects import hcl as hcl_d
from hcl_mlir.dialects import arith as arith_d
from hcl_mlir.dialects import memref as memref_d
from heterocl.context import get_context, get_location
import heterocl.ir.transform as hir


@hcl.register_primitive()
class BufferRootPrimitive(hcl.Primitive):
    name = "buffer_root"

    @staticmethod
    def apply(sch):
        mod = sch.module
        i32 = IntegerType.get_signless(32)
        for op in mod.body.operations[0].body.blocks[0].operations:
            if isinstance(op, memref_d.AllocOp):
                with op.location:
                    with InsertionPoint(op):
                        hcl_mlir.GlobalInsertionPoint.save(InsertionPoint(op))
                        cst = hcl_mlir.ConstantOp(i32, 0)
                        arith_d.AddIOp(cst.result, cst.result)


@hcl.register_primitive()
class AnnotatePrimitive(hcl.Primitive):
    name = "annotate"

    @staticmethod
    def apply(sch):
        loops = hir.get_affine_loop_nests(sch.top_func)
        for i, (name, loop) in enumerate(loops):
            hir.annotate(loop, f"Loop_{i}")


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
    print(s.module)
    hcl.build(s, target="vhls")
    print(s.module)


if __name__ == "__main__":
    test_gemm_buffer()
