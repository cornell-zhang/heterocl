import heterocl as hcl
import math as mt
import numpy as np

def top_cholesky(N, dtype=hcl.Int(), target=None):

    hcl.init(dtype)
    A = hcl.placeholder((N, N), "A")

    def kernel_cholesky(A):

        with hcl.Stage("loop_1"):
            with hcl.for_(0, N, name="i") as i:
                # Case: j < i
                with hcl.for_(0, i, name="j") as j:
                    with hcl.for_(0, j, name="k") as k:
                        A[i][j] = A[i][j] -  A[i][k] * A[j][k]
                    A[i][j] = A[i][j] / A[j][j]
                # Case: i == j
                with hcl.for_(0, i, name="k") as k:
                    A[i][i] = A[i][i] -  A[i][k] * A[i][k]
                A[i][i] = hcl.sqrt(A[i][i] * 1.0)
        
    s = hcl.create_schedule([A], kernel_cholesky)

    #### Apply customizations ####
    
    loop_1 = kernel_cholesky.loop_1

    #### Apply customizations ####

    return hcl.build(s, target=target)


def cholesky_golden(N, A):
    
    for i in range(N):
        for j in range(i):
            for k in range(j):
                A[i][j] -= A[i][k] * A[j][k]
            A[i][j] /= A[j][j]

        for k in range(i):
            A[i][i] -= A[i][k] * A[i][k]

        A[i][i] = mt.sqrt(A[i][i])

def main(N=32, dtype=hcl.Float(32), target=None):
    A = np.random.randint(10, size=(N, N)).astype(np.float32)
    f = top_cholesky(N, dtype, target)
    f(A)
    A_golden = np.zeros(A.shape, dtype=np.float32)
    cholesky_golden(N, A)
    if np.allclose(A, A_golden):
        print("pass")
    else:
        print("failed")

if __name__ == "__main__":
    main()