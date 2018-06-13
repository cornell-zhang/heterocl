import heterocl as hcl
import numpy as np

def popcount(a): # a is a 32-bit integer
  out = hcl.local(0)
  with hcl.for_(0, 32) as i:
    out[0] += a[i]
  return out[0]

A = hcl.placeholder((10, 10))
B = hcl.compute((10, 10), lambda x, y: popcount(A[x, y]))

s = hcl.create_schedule(B)
f = hcl.build(s, [A, B])

hcl_A = hcl.asarray(np.random.randint(32, size = A.shape), dtype = hcl.Int())
hcl_B = hcl.asarray(np.zeros(B.shape), dtype = hcl.Int())

f(hcl_A, hcl_B)

print hcl_A
print "=== hcl.compute ==="
print hcl_B


def popcount(A, B): # a is a 32-bit integer
  with hcl.for_(0, A.shape[0]) as x:
    with hcl.for_(0, A.shape[1]) as y:
      B[x, y] = 0
      with hcl.for_(0, 32) as i:
        B[x, y] += A[x, y][i]

A = hcl.placeholder((10, 10))
B = hcl.placeholder(A.shape)
with hcl.stage() as C:
  popcount(A, B)

s = hcl.create_schedule(C)
f = hcl.build(s, [A, B])

hcl_B = hcl.asarray(np.zeros(B.shape), dtype = hcl.Int())

f(hcl_A, hcl_B)

print "=== hcl.block ==="
print hcl_B
