import heterocl as hcl
import numpy as numpy
import tvm 

shape = (2,2)
dtype = "int32"

def increment_extern(A):
  with hcl.CodeBuilder() as cb:
    test = hcl.local(A)
    # test[0] = 6
    test[0] = test[0][5:2]
    # test[0] = test[0][0]
    return test[0]

def hcl_test_add():
  A = hcl.placeholder(shape, name = "A")
  #B = hcl.compute(shape, [A], lambda x, y: increment_extern(A, x, y), name = "B")
  B = hcl.compute(shape, [A], lambda x, y: increment_extern(A[x][y]))
  # dist = hcl.compute(diff.shape, [diff], lambda x, y: popcount(diff[x][y]))
  s = hcl.create_schedule(B)
  return hcl.build(s, [A, B])

# def hcl_test_external_inc():
#   A = hcl.placeholder((shape), name = "A")
#   B = hcl.block([A], increment_extern, name = "B")
#   s = hcl.create_schedule(B)
#   return hcl.build(s, [A,B])

_A = tvm.nd.array(numpy.array([[13,13],[13,13]]).astype(dtype))
_B = tvm.nd.array(numpy.zeros(shape).astype(dtype))
# hcl_test_external_inc()(_A, _B)
hcl_test_add()(_A,_B)
print _A
print _B
#print _C
