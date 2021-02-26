import heterocl as hcl
import os
import numpy as np

def test_aws_runtime(dtype=hcl.Int()):
    hcl.init(dtype)
    A = hcl.placeholder((2, 3), "A", dtype)
    B = hcl.placeholder((2, 3), "B", dtype)

    def kernel_gemm(A, B):
        return hcl.compute((2, 3),
                lambda x, y: A[x, y] * B[x, y], dtype = dtype, name = "C")

    s = hcl.create_schedule([A, B], kernel_gemm)
    target = hcl.Platform.aws_f1
    target.config(compile="vitis", mode="hw_sim")
    f = hcl.build(s, target=target)

    # Requires AWS CLI package
    if os.system("which aws >> /dev/null") != 0:
        return

    np_A = np.random.randint(10, size=(2,3))
    np_B = np.random.randint(10, size=(3,5))
    np_C = np.zeros((2,5))
    args = (np_A, np_B, np_C)

    # Generate local projects
    f.inspect(args)

    # Compile FPGA binary
    key_path = "/home/sx233/aws-sx233-test.pem"
    f.compile(args, remote=True, aws_key_path=key_path)

    # Execute bitstream on board
    f.execute(args, remote=True, aws_key_path=key_path)

if __name__ == "__main__":
    test_aws_runtime()