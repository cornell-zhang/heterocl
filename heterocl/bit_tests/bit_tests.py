import heterocl as hcl
import numpy as numpy
import tvm 
#def increment_extern(A, x, y):
#        return A[x][y] + 1 
#
#A = hcl.placeholder((10,), name = "A")
#B = hcl.block([A], increment, name = "B")
#
#s = hcl.create_schedule(B)

#A = tvm.placeholder((10,), name = "A")
#B = tvm.compute(A.shape, lambda x: A[x] + 1, name = "B")

shape = (2,2)
dtype = "int32"

def hcl_test_add():
  A = hcl.placeholder(shape, name = "A")
  #B = hcl.placeholder(shape, name = "B")
  #C = hcl.compute(shape, [A], lambda x, y: increment_extern(A, x, y), name = "C")
  C = hcl.compute(shape, [A], lambda x, y: A[x][y], name = "C")
  s = tvm.create_schedule(C.op)
  return hcl.build(s, [A, C])

#_A = tvm.nd.array(numpy.ones(shape).astype(dtype))
_A = tvm.nd.array(numpy.array([[0,0],[0,0]]).astype(dtype))
_B = tvm.nd.array(numpy.zeros(shape).astype(dtype))
_C = tvm.nd.array(numpy.zeros(shape).astype(dtype))
hcl_test_add()(_A,_C)
print _A
#print _B
print _C
