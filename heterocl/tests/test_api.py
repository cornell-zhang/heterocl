import heterocl as hcl
import numpy as np

"""
Testing API: make schedule
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
Testing API: make_scheme & make_schedule_from_scheme
"""

def test_resize():

  def algorithm(A, B):
    C = hcl.compute(A.shape, lambda x: A[x] + 1, "C")
    return hcl.update(B, lambda x: C[x] + 1)

  A = hcl.placeholder((10,), dtype = hcl.UInt(32))
  B = hcl.placeholder((10,), dtype = hcl.UInt(2))

  scheme = hcl.make_scheme([A, B], algorithm)
  scheme.downsize(algorithm.C, hcl.UInt(2))
  s = hcl.make_schedule_from_scheme(scheme)
  f = hcl.build(s, [A, B])

  a = np.random.randint(100, size=(10,))
  _A = hcl.asarray(a, dtype = hcl.UInt(32))
  _B = hcl.asarray(np.zeros(10), dtype = hcl.UInt(2))

  f(_A, _B)

  _A = _A.asnumpy()
  _B = _B.asnumpy()

  for i in range(10):
    assert(_B[i] == (a[i] + 2)%4)

"""
Testing API: unpack
"""
def test_unpack():

  def unpack(A, B):
    C = hcl.unpack(A, factor = 4, name = "C")
    U = hcl.update(B, lambda x: C[x])

    return U

  for i in range(4, 36, 4):
    A = hcl.placeholder((10,), "A", dtype = hcl.UInt(i))
    B = hcl.placeholder((40,), "B", dtype = hcl.UInt(i/4))

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

"""
Testing API: pack
"""
def test_pack():

  def pack(A, B):
    C = hcl.pack(A, factor = 4, name = "C")
    return hcl.update(B, lambda x: C[x])

  for i in range(4, 36, 4):
    A = hcl.placeholder((40,), "A", dtype = hcl.UInt(i/4))
    B = hcl.placeholder((10,), "B", dtype = hcl.UInt(i))

    s = hcl.make_schedule([A, B], pack)
    f = hcl.build(s, [A, B])

    _A = hcl.asarray(np.random.randint(1000, size = (40,)), dtype = hcl.UInt(i/4))
    _B = hcl.asarray(np.zeros(10), dtype = hcl.UInt(i))

    f(_A, _B)

    __A = _A.asnumpy()
    __B = _B.asnumpy()

    for j in range(0, 10):
      golden = 0
      numB = __B[j]
      for k in range(0, 4):
        numA = __A[j*4 + k]
        golden += numA << (k * i/4)
      assert numB == golden

def test_pack_unpack():

  def pack_unpack(A, B):
    C = hcl.pack(A, factor = 4)
    D = hcl.unpack(C, factor = 4)
    return hcl.update(B, lambda x: D[x])

  for i in range(1, 16):
    A = hcl.placeholder((40,), "A", dtype = hcl.UInt(i))
    B = hcl.placeholder((40,), "B", dtype = hcl.UInt(i))

    s = hcl.make_schedule([A, B], pack_unpack)
    f = hcl.build(s, [A, B])

    _A = hcl.asarray(np.random.randint(1000, size = (40,)), dtype = hcl.UInt(i))
    _B = hcl.asarray(np.zeros(40), dtype = hcl.UInt(i))

    f(_A, _B)

    __A = _A.asnumpy()
    __B = _B.asnumpy()

    for j in range(0, 40):
      assert __A[j] == __B[j]


