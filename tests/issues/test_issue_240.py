import heterocl as hcl
import os

def test_partition():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 

    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda x, y: A[x][y] + 1, "B")
        C = hcl.compute(A.shape, lambda x, y: B[x][y] + 1, "C") # add this line
        return C
    s = hcl.create_schedule(A, kernel)
    s.partition(kernel.B)
    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="debug")

    print(hcl.build(s, target))

if __name__ == "__main__":
    test_partition()
