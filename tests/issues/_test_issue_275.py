import heterocl as hcl
import numpy as np

def test_vitis():
    A = hcl.placeholder((10,), "A")
    W = hcl.placeholder((10,), "W")

    def kernel(A, W):
        B = hcl.compute(A.shape, 
                lambda i: A[i] + 1, "B")
        C = hcl.compute(B.shape,
                lambda i: B[i] + W[i], "C")
        return C

    target = hcl.Platform.aws_f1
    target.config(compiler="vitis", mode="debug", project="project-vitis.prj")
    s = hcl.create_schedule([A, W], kernel)
    s.to([A, W], target.xcel)
    s.to(kernel.C, target.host)
    s.to(kernel.B, s[kernel.C])
    f = hcl.build(s, target)
    np_A = np.zeros((10,))
    np_W = np.zeros((10,))
    np_C = np.zeros((10,))
    hcl_A = hcl.asarray(np_A)
    hcl_W = hcl.asarray(np_W)
    hcl_C = hcl.asarray(np_C)
    assert "bundle=gmem0" in f
    assert "bundle=gmem1" in f
    assert "bundle=gmem2" in f

