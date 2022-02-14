import heterocl as hcl

def test_hls_function_array_interface():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
        C = hcl.compute(A.shape, lambda *args : B[args] + 1, "C")
        D = hcl.compute(A.shape, lambda *args : C[args] * 2, "D")
        return D
    
    target = hcl.Platform.aws_f1
    s = hcl.create_schedule([A], kernel)
    target.config(compiler="vivado_hls", mode="debug")
    f = hcl.build(s, target)
    assert "void test(int A[10][32], int D[10][32])" in f

