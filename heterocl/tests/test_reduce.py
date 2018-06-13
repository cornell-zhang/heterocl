import heterocl as hcl
import tvm
import numpy

import pytest

shape_in = (10, 10)
shape_out = (10,)

dtype = "int32"

def np_sum(A):
  return numpy.sum(A, axis = 1)

def hcl_test_sum():
  A = hcl.placeholder(shape_in, name = "A")
  k = hcl.reduce_axis(0, 10, name = "k")
  B = hcl.compute(shape_out, lambda x: hcl.sum(A[x, k], k), name = "B")
  s = hcl.create_schedule(B)
  print hcl.lower(s, [A, B])
  return hcl.build(s, [A, B])


@pytest.mark.parametrize("hcl_func, numpy_func, assertion", [
  (hcl_test_sum, np_sum, 0),
  ])
def test_compute(hcl_func, numpy_func, assertion):
  _A = numpy.random.randint(100, size = shape_in).astype(dtype)
  _B = numpy.zeros(shape_out).astype(dtype)

  __A = tvm.nd.array(_A)
  __B = tvm.nd.array(_B)

  try:
    hcl_func()(__A, __B)
    _B = numpy_func(_A)

    numpy.testing.assert_allclose(__B.asnumpy(), _B, rtol=1e-5)
  except AssertionError as e:
    print str(e)
    assert assertion==1


