import heterocl as hcl
import numpy as numpy
import tvm 

def insert_sort(A, B):
  for i in range(1, 10):
    for j in range(i, 0, -1):
      if A[j] > A[j-1]:
        swap(A[j], A[j-1])
  for k in range(0, 10):
    B[k] = A[k]

A = hcl.placeholder((10,), name = "A")
B = hcl.block([A], insert_sort, name = "B")

s = hcl.create_schedule(B)
s[B].pipeline(B.loops[0])