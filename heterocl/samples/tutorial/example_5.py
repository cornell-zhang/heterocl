import heterocl as hcl
import numpy as np

def popcount(n):
  l = hcl.local(0)
  with hcl.for_(0, 32) as i:
    l[0] += n[i]
  return l[0]

f = hcl.function([1], lambda n: popcount(n), False, name = "popcount")

m = hcl.var("m")
A = hcl.placeholder((10,), "A")
B = hcl.compute(A.shape, lambda x: f(A[x]), "B")
C = hcl.compute(A.shape, lambda x: f(A[x] & m), "C")

s = hcl.create_schedule([B, C])
stmt = hcl.lower(s, [m, A, B, C])
f = hcl.build(s, [m, A, B, C])

m = 0b1
_A = hcl.asarray(np.random.randint(50, size = (10,)), hcl.Int())
_B = hcl.asarray(np.zeros((10,)), hcl.Int())
_C = hcl.asarray(np.zeros((10,)), hcl.Int())

f (m, _A, _B, _C)

print _A
print _B
print _C
