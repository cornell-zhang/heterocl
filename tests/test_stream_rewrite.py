import heterocl as hcl
import numpy as np
from itertools import permutations
import os

target = hcl.platform.aws_f1

def test_extract_subgraph():

    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")
    C = hcl.placeholder((10, 32), "C")
    D = hcl.compute(A.shape, lambda y, x: A[y, x] + B[y, x], "D")
    E = hcl.compute(C.shape, lambda y, x: C[y, x] * D[y, x], "E")
    F = hcl.compute((10, 30), lambda y, x: E[y, x] + E[y, x+1] + E[y, x+2], "F")

    s = hcl.create_schedule([A, B, C, D, E, F])
    RB = s.reuse_at(E, s[F], F.axis[1])
    s.partition(RB, hcl.Partition.Block)

    # create super stage for sub-graphs
    s.to([A, B, C], target.xcel)
    s.to(E, target.host)

    # create new sch and return top stage 
    # top = s.graph()
    # top.dataflow()

    code = str(hcl.lower(s))
    print(code)

def test_super_stage():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")

    def kernel(A, B):
        C = hcl.compute((10, 32), lambda *args : 0, "C")

        with hcl.Stage("Super") as m:
            hcl.update(C, lambda *args: C[args] + 1, "update")

            with hcl.Stage("Plus") as stage:
                with hcl.for_(0, 10) as j:
                    C[j, 0] = 10
        return C

    s = hcl.create_schedule([A, B], kernel)
    s.to([A, B], target.xcel)
    s.to(kernel.Super.C, target.host)

if __name__ == '__main__':
    test_extract_subgraph()
