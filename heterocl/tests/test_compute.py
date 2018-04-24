import heterocl as hcl
import tvm
import numpy

import pytest

shape = (10, 10)
dtype = "int32"

def add(A, B, C):
  for i in range(0, 10):
    for j in range(0, 10):
      C[i][j] = A[i][j] + B[i][j]

def add_extern(A, B, x, y):
  return A[x][y] + B[x][y]

def hcl_test_add():
  A = hcl.placeholder(shape, name = "A")
  B = hcl.placeholder(shape, name = "B")
  C = hcl.compute(shape, [A, B], lambda x, y: A[x][y] + B[x][y], name = "C")
  s = hcl.create_schedule(C)
  return hcl.build(s, [A, B, C])

def hcl_test_add_extern():
  A = hcl.placeholder(shape, name = "A")
  B = hcl.placeholder(shape, name = "B")
  C = hcl.compute(shape, [A, B], lambda x, y: add_extern(A, B, x, y), name = "C")
  s = hcl.create_schedule(C)
  return hcl.build(s, [A, B, C])


@pytest.mark.parametrize("hcl_func, numpy_func, assertion", [
  (hcl_test_add, add, 0),
  (hcl_test_add_extern, add, 0)
  ])
def test_compute(hcl_func, numpy_func, assertion):
  _A = numpy.random.randint(100, size = shape).astype(dtype)
  _B = numpy.random.randint(100, size = shape).astype(dtype)
  _C = numpy.zeros(shape).astype(dtype)

  __A = tvm.nd.array(_A)
  __B = tvm.nd.array(_B)
  __C = tvm.nd.array(_C)

  try:
    hcl_func()(__A, __B, __C)
    numpy_func(_A, _B, _C)

    numpy.testing.assert_allclose(__C.asnumpy(), _C, rtol=1e-5)
  except AssertionError as e:
    print str(e)
    assert assertion==1


