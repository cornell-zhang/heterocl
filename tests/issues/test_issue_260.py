import heterocl as hcl
import numpy as np
import os

def test_host_dtype():
    if os.system("which v++ >> /dev/null") != 0:
        return 
    qtype = hcl.Fixed(16,12)
    
    A = hcl.placeholder((10,), "A", dtype=qtype)
    def kernel(A):
        return hcl.compute((10,), lambda x: A[x] + 1, "B", dtype=qtype)

    s = hcl.create_schedule(A, kernel)
    target = hcl.Platform.aws_f1
    target.config(compile="vitis", mode="sw_sim", backend="vhls")
    f = hcl.build(s, target=target)
    hcl_A = hcl.asarray(np.random.randint(0, 10, A.shape), dtype=qtype)
    hcl_B = hcl.asarray(np.random.randint(0, 10, A.shape), dtype=qtype)
    f(hcl_A, hcl_B)

if __name__ == '__main__':
    test_host_dtype()
