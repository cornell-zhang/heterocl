import heterocl as hcl
import numpy as np

a = hcl.var()
A = hcl.placeholder((10, 10))
B = hcl.compute(A.shape, lambda x, y: A[x, y] * a)

s = hcl.create_schedule(B)
f = hcl.build(s, [a, A, B])

hcl_a = 10
hcl_A = hcl.asarray(np.random.randint(100, size = A.shape), dtype = hcl.Int())
hcl_B = hcl.asarray(np.zeros(B.shape), dtype = hcl.Int())


f(hcl_a, hcl_A, hcl_B)

print hcl_a
print hcl_A.asnumpy()
print hcl_B.asnumpy()

