import heterocl as hcl
import numpy as np

def test_pipe():
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda i: A[i] + 1, "B", 
                module=True, inputs=[A])
        C = hcl.compute((11,), lambda i: hcl.select(i > 0, B[i], 0),"C",
                module=True, inputs=[B])
        return C

    target = hcl.Platform.zc706
    target.config(compile="vivado_hls", mode="csyn")
    s = hcl.create_schedule([A], kernel)

    s.to([A], target.xcel)
    s.to(kernel.C, target.host)
    s.to(kernel.B, s[kernel.C])

    print(hcl.lower(s))

if __name__ == '__main__':
    test_pipe()
