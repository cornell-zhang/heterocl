import heterocl as hcl
import numpy as np
import os

def test_imperative():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    dtype = hcl.Float()
    A = hcl.placeholder((4, 4), "A", dtype)

    def kernel(A):
        def func(data):
            out = hcl.compute((4, 4),lambda x, y: 0, "out", dtype)
            with hcl.Stage("S"):
                with hcl.for_(0, 4, name="i") as i:
                    with hcl.for_(0, 4, name="j") as j:
                        out[i, j] = data[i, j] + 1
            return out
        B = func(A)
        C = hcl.compute((4,4), lambda i, j: B[i, j] + 1, "C")
        return C

    s = hcl.create_schedule([A], kernel)

    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="csyn")
    s.to(A, target.xcel)
    s.to(kernel.C, target.host)
    f = hcl.build(s, target=target)
    np_A = np.random.randint(0, 10, A.shape)
    hcl_A = hcl.asarray(np_A,dtype)
    hcl_B = hcl.asarray(np.zeros((4, 4),np.float),dtype)
    f(hcl_A, hcl_B)


if __name__ == "__main__":
    test_imperative()
