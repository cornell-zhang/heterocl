import heterocl as hcl

def test_host_device_split():
    target = hcl.platform.aws_f1

    def test_placeholders():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        B = hcl.placeholder((10, 32), "B")
        C = hcl.placeholder((10, 32), "C")
        D = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], "D")
        E = hcl.compute(C.shape, lambda i, j: C[i][j] * D[i][j], "E")
        F = hcl.compute(C.shape, lambda i, j: E[i][j] + 1, "F")

        s = hcl.create_schedule([A, B, C, D, E, F])
        s.to([A, B, C], target.xcel)
        s.to(E, target.host)
        code = str(hcl.lower(s))
        assert "top_function_0" in code
        assert "top_function_1" not in code

    def test_extern_ops():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        def kernel(A):
            B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
            C = hcl.compute(A.shape, lambda *args : B[args] + 1, "C")
            D = hcl.compute(A.shape, lambda *args : C[args] * 2, "D")
            return D
        
        s = hcl.create_schedule([A], kernel)
        s.to(kernel.B, target.xcel)
        s.to(kernel.C, target.host)
        code = str(hcl.lower(s))
        assert "top_function_0" in code
        assert "top_function_1" not in code

    test_placeholders()
    test_extern_ops()

