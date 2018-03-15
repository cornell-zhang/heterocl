import hcl

def myfun(A):
  return A[2, 4]

A = hcl.placeholder((10, 10), name = "A")
B = hcl.compute((10, 10), [A], lambda x, y: A[x][y] + 1, inline = False)


