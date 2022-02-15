import heterocl as hcl
import numpy as np

hcl.init()

A = hcl.placeholder((10,))

def kernel(A):
    hcl.print(A)

s = hcl.create_schedule([A], kernel)
f = hcl.build(s)

np_A = np.random.randint(0, 10, size=(10,))
hcl_A = hcl.asarray(np_A)

f(hcl_A)

s = "["
for i in range(0, 10):
    s += str(np_A[i])
    if i < 9:
        s += ", "
s += "]"
print(s)
