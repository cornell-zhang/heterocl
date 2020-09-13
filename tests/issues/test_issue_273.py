import numpy as np
import heterocl as hcl

def test_conv2():
    if os.system("which vivado_hls >> /dev/null") != 0:
        return 
    A = hcl.placeholder((6, 6), "A")
    F = hcl.placeholder((3, 3), "F")

    def kernel(A, F):
        r = hcl.reduce_axis(0, 3)
        c = hcl.reduce_axis(0, 3)
        return hcl.compute((4, 4),
                lambda y, x: hcl.sum(A[y+r, x+c] * F[r, c], axis=[r, c]), "B")

    s = hcl.create_schedule([A, F], kernel)
    LB = s.reuse_at(A, s[kernel.B], kernel.B.axis[0], "LB")
    WB = s.reuse_at(LB, s[kernel.B], kernel.B.axis[1], "WB")
    s.partition(F, hcl.Partition.Cyclic, factor=3, dim=2)
    s.partition(F, hcl.Partition.Cyclic, factor=3, dim=1) # different dimensions
    s[kernel.B].pipeline(kernel.B.axis[1])
    
    target = hcl.platform.zc706
    target.config(compile="vivado_hls",mode="csyn")
    f = hcl.build(s, target=target)
    hcl_A = hcl.asarray(np.random.randint(0, 10, A.shape))
    hcl_F = hcl.asarray(np.random.randint(0, 10, F.shape))
    hcl_B = hcl.asarray(np.zeros((4, 4)))
    f(hcl_A, hcl_F, hcl_B)

if __name__ == '__main__':
    test_conv2()
