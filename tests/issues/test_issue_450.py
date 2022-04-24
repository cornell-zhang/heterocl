import heterocl as hcl
import numpy as np

def test_mask():

    def two_stage(A):
        var = hcl.scalar(0, "v", dtype=hcl.UInt(32))
        var.v = 1
        with hcl.if_(var == 0):
            hcl.print((),"A\n")
        with hcl.else_():
            var.v = var - 1
            # this condition should not be optimized away
            with hcl.if_(var == 0):
                hcl.print((),"B\n")
        A[0] = var
        return A

    A = hcl.placeholder((2,), "A", dtype=hcl.UInt(16))
    s = hcl.create_schedule([A], two_stage)
    print(hcl.lower(s))

if __name__ == "__main__":
    test_mask()
