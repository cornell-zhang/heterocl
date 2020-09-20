import heterocl as hcl
import numpy as np

def test_pipe():
    A = hcl.placeholder((10,), "A")

    def kernel(A):
        B = hcl.compute(A.shape, 
                lambda i: A[i] + 1, "B")
        C = hcl.compute((11,),
                lambda i: hcl.select(i > 0, B[i], 0),"C")
        return C

    target = hcl.platform.zc706
    target.config(compile="vivado_hls", mode="csyn")
    s = hcl.create_schedule([A], kernel)
    print(hcl.lower(s))

    s.to([A], target.xcel)
    s.to(kernel.C, target.host)
    s.to(kernel.B, s[kernel.C])

    f = hcl.build(s, target)
    np_A = np.zeros((10,))
    np_C = np.zeros((11,))
    hcl_A = hcl.asarray(np_A)
    hcl_C = hcl.asarray(np_C)
    f(hcl_A, hcl_C)

if __name__ == '__main__':
    test_pipe()
