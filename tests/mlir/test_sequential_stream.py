import heterocl as hcl
import numpy as np

def test_stages():

    A = hcl.placeholder((32, 32), "A")
    C = hcl.placeholder((32, 32), "C")
    def kernel(A, C):
        B = hcl.compute(A.shape, lambda i, j : A[i, j] + 1, "B")
        D = hcl.compute(A.shape, lambda i, j : B[i, j] + 1, "D")
        E = hcl.compute(A.shape, lambda i, j : C[i, j] + 1, "E")
        F = hcl.compute(A.shape, lambda i, j : D[i, j] + E[i, j], "F")
        return F

    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="csyn", project="stages-tvm-seq.prj")
    s = hcl.create_schedule([A, C], kernel)
    s.to(kernel.B, s[kernel.D])
    s.to(kernel.D, s[kernel.F])
    s.to(kernel.E, s[kernel.F])
    mod = hcl.build(s, target=target)
    np_A = np.zeros((32, 32))
    np_C = np.zeros((32, 32))
    np_F = np.zeros((32, 32))
    hcl_A = hcl.asarray(np_A)
    hcl_C = hcl.asarray(np_C)
    hcl_F = hcl.asarray(np_F)
    mod(hcl_A, hcl_C, hcl_F)
    report = mod.report()
    report.display()
    

if __name__ == "__main__":
    test_stages()