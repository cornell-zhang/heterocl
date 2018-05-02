import heterocl as hcl
import numpy as np
import tvm

a = hcl.var()
A = hcl.placeholder((10,))
B = hcl.compute((10,), [A], lambda x: A[x] * a)

hcl.resize([a, A], "uint5")
hcl.resize(B, "uint10")

s = hcl.create_schedule(B)

print tvm.lower(s, [a.var, A.tensor, B.tensor], simple_mode = True)

f = hcl.build(s, [a, A, B])

hcl_a = 10
hcl_A = hcl.asarray(np.random.randint(31, size = (10,)), dtype = "uint5")
hcl_B = hcl.asarray(np.zeros((10,)), dtype = "uint10")


f(hcl_a, hcl_A, hcl_B)

print hcl_A.asnumpy()
print hcl_B.asnumpy()

