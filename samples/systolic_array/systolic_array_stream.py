import heterocl as hcl
import numpy as np
import time
from systolic_array_main import gemm

m = k = n = 16
x_max = y_max = 16
dtype = hcl.Int()

def systolic(m=16, k=16, n=16, dtype=hcl.Int(), target=None):
    hcl.init(dtype)

    dim_x, dim_y = 16, 16
    A  = hcl.placeholder((m, k), dtype=dtype, name="A")
    B  = hcl.placeholder((k, n), dtype=dtype, name="B")
    output = hcl.placeholder((m, n), dtype=dtype, name="output")

    def kernel(A, B, O):

        stream_A = hcl.compute((m, k-1), lambda *args: 0, "stream_A")
        stream_B = hcl.compute((k-1, n), lambda *args: 0, "stream_B")

        def update(k, y, x):
            last = hcl.scalar(
                hcl.select(k==0, 0, O[y, x]), "last")

            stream_A[y, x] = hcl.select(x>0, stream_A[y, x-1], A[y, k])
            stream_B[y, x] = hcl.select(y>0, stream_B[y-1, x], B[k, x])
            O[y, x] = last.v + stream_A[y, x] * stream_B[y, x]

        hcl.mutate((m, dim_y, dim_x), 
            lambda k, y, x: update(k, y, x), name="update")

    s = hcl.create_schedule([A, B, output], kernel)

    k = kernel.update
    s[k].pipeline(k.axis[0])
    
    s.to(k.stream_A, s[k])
    s.to(k.stream_B, s[k])

    print(hcl.lower(s))
    f = hcl.build(s, target=target)
    return f
    
np_1 = np.random.randint(10, size=(m, k))
np_2 = np.random.randint(10, size=(k, n))
np_3 = np.matmul(np_1, np_2)

hcl_m1 = hcl.asarray(np_1, dtype=dtype)
hcl_m2 = hcl.asarray(np_2, dtype=dtype)
hcl_m3 = hcl.asarray(np.zeros((m, n)), dtype=dtype)
hcl_m4 = hcl.asarray(np.zeros((m, n)), dtype=dtype)

fg = gemm(m, n, k, dtype=hcl.Int(), target="llvm")
fs = systolic(m, n, k, dtype=hcl.Int(), target="llvm")

fg(hcl_m1, hcl_m2, hcl_m3)
fs(hcl_m1, hcl_m2, hcl_m4)
assert np.array_equal(hcl_m3.asnumpy(), hcl_m4.asnumpy())
