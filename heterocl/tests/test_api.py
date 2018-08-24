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

"""
Testing API: unpack
"""
def test_unpack():
  A = hcl.placeholder((10,), "A")
  B = hcl.placeholder((40,), "B")

  def unpack(A, B):
    C = hcl.unpack(A, factor = 4, name = "C")
    U = hcl.update(B, lambda x: C[x])

    return U

  for i in range(4, 36, 4):
    [A] = hcl.downsize(A, hcl.UInt(i))
    [B] = hcl.downsize(B, hcl.UInt(i/4))

    s = hcl.make_schedule([A, B], unpack)
    f = hcl.build(s, [A, B])

    _A = hcl.asarray(np.random.randint(1000, size = (10,)), dtype = hcl.UInt(i))
    _B = hcl.asarray(np.zeros(40), dtype = hcl.UInt(i/4))

    f(_A, _B)

    __A = _A.asnumpy()
    __B = _B.asnumpy()

    for j in range(0, 10):
      for k in range(0, 4):
        numA = __A[j]
        numB = __B[j*4 + k]
        golden = (numA >> (i/4*k)) % (1 << (i/4))
        assert numB == golden

