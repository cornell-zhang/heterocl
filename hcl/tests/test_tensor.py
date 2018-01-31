import hcl

def myfun(A):
  return A[2, 4]

A = hcl.placeholder((10, 10), "A")
B = hcl.compute((2, 2), lambda x, y: myfun(A) + 1)


