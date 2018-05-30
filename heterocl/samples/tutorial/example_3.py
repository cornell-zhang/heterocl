import heterocl as hcl
import numpy as np

def shift_op(A, x, k):
  with hcl.if_(k == 9):
    A[k] = 0
  with hcl.else_():
    A[k] = A[k + 1]

A = hcl.placeholder((10,))
B = hcl.mut_compute((5, 10), [A], lambda x, k: shift_op(A, x, k))

s = hcl.create_schedule(B)
f = hcl.build(s, [A])

hcl_A = hcl.asarray(np.random.randint(20, size = A.shape), dtype = hcl.Int())

print hcl_A

f(hcl_A)

print hcl_A
