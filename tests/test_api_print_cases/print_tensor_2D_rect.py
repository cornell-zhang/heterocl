import heterocl as hcl
import numpy as np

hcl.init()

A = hcl.placeholder((5, 10))

def kernel(A):
    hcl.print(A)

s = hcl.create_schedule([A], kernel)
f = hcl.build(s)

np_A = np.random.randint(0, 10, size=(10, 10))
hcl_A = hcl.asarray(np_A)

f(hcl_A)

s = "["
for i in range(0, 5):
    s += "["
    for j in range(0, 10):
        s += str(np_A[i][j])
        if j < 9:
            s += ", "
    s += "]"
    if i < 4:
        s += ",\n"
s += "]"
print(s)
