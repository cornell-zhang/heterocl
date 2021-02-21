import heterocl as hcl

def test_redundancy(dtype=hcl.Int()):
    hcl.init(dtype)
    A = hcl.placeholder((2, 3), "A")
    B = hcl.placeholder((3, 5), "B")
    C = hcl.placeholder((2, 5), "C")

    def kernel_gemm(A, B, C):
        r = hcl.reduce_axis(0, 3, "r")
        out_AB = hcl.compute((2, 3),
                lambda x, y: hcl.sum(2 * A[x, r] * B[r, y],
                axis = r, dtype = dtype), name = "out_AB")
        hcl.update(C, lambda x, y: 3 * C[x, y] + out_AB[x, y], name = "C")

    s = hcl.create_schedule([A, B, C], kernel_gemm)
    code = hcl.build(s, target="vhls")
    print(code)

test_redundancy()