import heterocl as hcl
import os

def test_aws_runtime(dtype=hcl.Int()):
    hcl.init(dtype)
    A = hcl.placeholder((2, 3), "A")
    B = hcl.placeholder((3, 5), "B")
    C = hcl.placeholder((2, 5), "C")

    def kernel_gemm(A, B, C):
        r = hcl.reduce_axis(0, 3, "r")
        out_AB = hcl.compute((2, 3),
                lambda x, y: hcl.sum(2 * A[x, r] * B[r, y],
                axis = r, dtype = dtype), name = "out_AB")
        hcl.update(C, lambda x, y: 3 * C[x, y] + out_AB[x, y], name = "C1")

    s = hcl.create_schedule([A, B, C], kernel_gemm)
    target = hcl.Platform.aws_f1
    f = hcl.build(s, target=target)

    if os.system("which aws >> /dev/null") != 0:
        return 
    f.compile(remote=True, aws_key_path=None)

test_aws_runtime()