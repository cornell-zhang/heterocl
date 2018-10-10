import heterocl as hcl
import tvm
import numpy

import pytest

shape = (10, 10)

def copy(A, B, C, bit):
  for i in range(0, 10):
    for j in range(0, 10):
      C[i][j] = (A[i][j]) % (1 << bit)

def add(A, B, C, bit):
  for i in range(0, 10):
    for j in range(0, 10):
      C[i][j] = (A[i][j] + B[i][j]) % (1 << bit)

def hcl_test_copy(dtype):
  A = hcl.placeholder(shape, name = "A", dtype=dtype)
  B = hcl.placeholder(shape, name = "B", dtype=dtype)
  C = hcl.compute(shape, lambda x, y: A[x][y], name = "C", dtype=dtype)
  s = hcl.create_schedule(C)
  return hcl.build(s, [A, B, C])

def hcl_test_add(dtype):
  A = hcl.placeholder(shape, name = "A", dtype=dtype)
  B = hcl.placeholder(shape, name = "B", dtype=dtype)
  C = hcl.compute(shape, lambda x, y: A[x][y] + B[x][y], name = "C", dtype=dtype)
  s = hcl.create_schedule(C)
  return hcl.build(s, [A, B, C])

@pytest.mark.parametrize("hcl_func, numpy_func, bit, assertion", [
  (hcl_test_copy, copy, 8, 0),
  (hcl_test_copy, copy, 16, 0),
  (hcl_test_copy, copy, 32, 0),
  (hcl_test_copy, copy, 64, 0),
  (hcl_test_copy, copy, 1, 0),
  (hcl_test_copy, copy, 3, 0),
  (hcl_test_copy, copy, 4, 0),
  (hcl_test_copy, copy, 6, 0),
  (hcl_test_copy, copy, 13, 0),
  (hcl_test_copy, copy, 21, 0),
  (hcl_test_copy, copy, 29, 0),
  (hcl_test_copy, copy, 37, 0),
  (hcl_test_copy, copy, 46, 0),
  (hcl_test_copy, copy, 54, 0),
  (hcl_test_add, add, 8, 0),
  (hcl_test_add, add, 16, 0),
  (hcl_test_add, add, 32, 0),
  (hcl_test_add, add, 64, 0),
  (hcl_test_add, add, 1, 0),
  (hcl_test_add, add, 3, 0),
  (hcl_test_add, add, 4, 0),
  (hcl_test_add, add, 6, 0),
  (hcl_test_add, add, 13, 0),
  (hcl_test_add, add, 21, 0),
  (hcl_test_add, add, 29, 0),
  (hcl_test_add, add, 37, 0),
  (hcl_test_add, add, 46, 0),
  (hcl_test_add, add, 54, 0)
  ])
def test_apint(hcl_func, numpy_func, bit, assertion):
  dtype = "uint" + str(bit)
  _A = numpy.random.randint(1 << 62, size=shape)
  _B = numpy.random.randint(1 << 62, size=shape)
  _C = numpy.zeros(shape).astype("int")

  __A = tvm.nd.array(_A, dtype=dtype)
  __B = tvm.nd.array(_B, dtype=dtype)
  __C = tvm.nd.array(_C, dtype=dtype)

  try:
    hcl_func(dtype)(__A, __B, __C)
    numpy_func(_A, _B, _C, bit)

    numpy.testing.assert_allclose(__C.asnumpy(), _C, rtol=1e-5)
  except AssertionError as e:
    print str(e)
    assert assertion == 1

