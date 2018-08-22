import heterocl as hcl
import numpy as np

"""
Testing API: make_schedule
"""

def test_base():
  A = hcl.placeholder((10,))
  B = hcl.placeholder((10,))

  def algorithm(A, B):
    U = hcl.update(B, lambda x: A[x] + 1)
    return U

  s = hcl.make_schedule([A, B], algorithm)
  f = hcl.build(s, [A, B])

  _A = hcl.asarray(np.random.randint(100, size=(10,)), dtype = hcl.Int(32))
  _B = hcl.asarray(np.zeros(10), dtype = hcl.Int(32))

  f(_A, _B)

  _A = _A.asnumpy()
  _B = _B.asnumpy()

  for i in range(10):
    assert(_B[i] == _A[i] + 1)

def test_compute():
  A = hcl.placeholder((10,))
  B = hcl.placeholder((10,))

  def algorithm(A, B):
    C = hcl.compute(A.shape, lambda x: A[x] + 1)
    U = hcl.update(B, lambda x: C[x] + 1)
    return U

  s = hcl.make_schedule([A, B], algorithm)
  f = hcl.build(s, [A, B])

  _A = hcl.asarray(np.random.randint(100, size=(10,)), dtype = hcl.Int(32))
  _B = hcl.asarray(np.zeros(10), dtype = hcl.Int(32))

  f(_A, _B)

  _A = _A.asnumpy()
  _B = _B.asnumpy()

  for i in range(10):
    assert(_B[i] == _A[i] + 2)

def test_resize():
  A = hcl.placeholder((10,))
  B = hcl.placeholder((10,))

  def algorithm(A, B):
    C = hcl.compute(A.shape, lambda x: A[x] + 1)
    U = hcl.update(B, lambda x: C[x] + 1)
    return U

  [A] = hcl.downsize(A, hcl.UInt(2))
  s = hcl.make_schedule([A, B], algorithm)
  f = hcl.build(s, [A, B])

  a = np.random.randint(100, size=(10,))
  _A = hcl.asarray(a, dtype = hcl.UInt(2))
  _B = hcl.asarray(np.zeros(10), dtype = hcl.Int(32))

  f(_A, _B)

  _A = _A.asnumpy()
  _B = _B.asnumpy()

  for i in range(10):
    assert(_B[i] == a[i]%4 + 2)

def test_resize_with_branch():
  A = hcl.placeholder((10,))
  B = hcl.placeholder((10,))

  def algorithm(A, B):
    if (A.dtype == "int32"):
      U = hcl.update(B, lambda x: A[x] + 1)
      return U
    else:
      U = hcl.update(B, lambda x: A[x] + 2)
      return U

  # create two schedule
  def with_resize(A, B):
    [A] = hcl.downsize(A, hcl.UInt(2))
    s = hcl.make_schedule([A, B], algorithm)
    f = hcl.build(s, [A, B])

    a = np.random.randint(100, size=(10,))
    _A = hcl.asarray(a, dtype = hcl.UInt(2))
    _B = hcl.asarray(np.zeros(10), dtype = hcl.Int(32))

    f(_A, _B)

    _A = _A.asnumpy()
    _B = _B.asnumpy()

    for i in range(10):
      assert(_B[i] == a[i]%4 + 2)

  def without_resize(A, B):
    s = hcl.make_schedule([A, B], algorithm)
    f = hcl.build(s, [A, B])

    a = np.random.randint(100, size=(10,))
    _A = hcl.asarray(a, dtype = hcl.Int(32))
    _B = hcl.asarray(np.zeros(10), dtype = hcl.Int(32))

    f(_A, _B)

    _A = _A.asnumpy()
    _B = _B.asnumpy()

    for i in range(10):
      assert(_B[i] == a[i] + 1)

  with_resize(A, B)
  without_resize(A, B)

def test_schedule():
  A = hcl.placeholder((10,))
  B = hcl.placeholder((10,))

  def algorithm(A, B):
    U = hcl.update(B, lambda u0: A[u0] + 1, name="U")
    return U

  s = hcl.make_schedule([A, B], algorithm)
  s[algorithm.U].unroll(algorithm.U.axis[0])

  s = hcl.lower(s, [A, B])

  assert 'unrolled "factor"=0 (u0, 0, 10)' in str(s)
