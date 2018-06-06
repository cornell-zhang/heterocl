import heterocl as hcl
import numpy as np

k1 = hcl.reduce_axis(0, 10, "k1")
k2 = hcl.reduce_axis(0, 10, "k2")

A = hcl.placeholder((10, 10), "A")
B = hcl.compute((1,), [A], lambda x: hcl.sum(A[k1, k2], axis = [k1, k2], where = A[k1, k2] > 0), "B")

s = hcl.create_schedule(B)
f = hcl.build(s, [A, B])

_A = hcl.asarray(np.random.randint(-10, 10, size = A.shape), hcl.Int())
_B = hcl.asarray(np.zeros(B.shape), hcl.Int())

f(_A, _B)

print _A
print _B
