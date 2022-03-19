import heterocl as hcl
import numpy as np

A = hcl.placeholder((10,))

hcl.init()

def kernel(A):
    hcl.print(5)
    hcl.print(2.5)

s = hcl.create_schedule([A], kernel)
f = hcl.build(s)

np_A = np.random.rand(10)
hcl_A = hcl.asarray(np_A)

f(hcl_A)
