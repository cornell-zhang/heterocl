import heterocl as hcl
import numpy as np
import time

m = k = n = 16
x_max = y_max = 16
dtype = hcl.Int()
host_only = False

def systolic(m=16, k=16, n=16, dtype=hcl.Int(), target=None):
    hcl.init(dtype)

    dim_x, dim_y = 16, 16
    A  = hcl.placeholder((m, k), dtype=dtype, name="A")
    B  = hcl.placeholder((k, n), dtype=dtype, name="B")

    def kernel(A, B):

        localA = hcl.compute((m, k-1), lambda *args: 0, "localA")
        localB = hcl.compute((k-1, n), lambda *args: 0, "localB")
        output = hcl.compute((m, n), lambda *args: 0, "output")

        def update(k, y, x):

            localA[y, x] = hcl.select(x>0, localA[y, x-1], A[y, k])
            localB[y, x] = hcl.select(y>0, localB[y-1, x], B[k, x])
            output[y, x] = hcl.select(k==0, 0, output[y, x]) + localA[y, x] * localB[y, x]

        hcl.mutate((m, dim_y, dim_x), 
            lambda k, y, x: update(k, y, x), name="update")
        return output

    s = hcl.create_schedule([A, B], kernel)

    k = kernel.update
    s[k].pipeline(k.axis[0])
    
    # self loopback streaming 
    s.to(k.localA, kernel.update)
    s.to(k.localB, kernel.update)

    # move to xcel scope 
    if not host_only:
        s.to([A, B], target.xcel)
        s.to(k.output, target.host)

    print(hcl.lower(s))
    f = hcl.build(s, target=target)
    return f
    
np_1 = np.random.randint(10, size=(m, k))
np_2 = np.random.randint(10, size=(k, n))
np_3 = np.matmul(np_1, np_2)

hcl_m1 = hcl.asarray(np_1, dtype=dtype)
hcl_m2 = hcl.asarray(np_2, dtype=dtype)
hcl_m3 = hcl.asarray(np.zeros((m, n)), dtype=dtype)

target = hcl.platform.aws_f1
target.config(compile="vitis", backend="vhls")
fs = systolic(m, n, k, dtype=hcl.Int(), target=target)
fs(hcl_m1, hcl_m2, hcl_m3)

print(hcl_m3.asnumpy())
print(np.matmul(np_1, np_2))
