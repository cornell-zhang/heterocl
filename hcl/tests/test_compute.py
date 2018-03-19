import hcl
import tvm
import numpy

def foo(A, x, y):
  return A[x][y] + 1.0

def foo_np(a):
  func = numpy.vectorize(lambda x: x + 1)
  return func(a)

def foo2(A, x, y):
  out = 5.0 if A[x][y] > 0.5 else 10.0
  return out

def foo2_np(a):
  func = numpy.vectorize(lambda x: 5.0 if x > 0.5 else 10.0)
  return func(a)

def test0(A, a): # test basic hcl.compute
  B = hcl.compute(A.shape, [A], lambda x, y: A[x][y] + 1.0)
  b = foo_np(a)
  return B, b

def test1(A, a): # test basic compute with inline False
  B = hcl.compute(A.shape, [A], lambda x, y: A[x][y] + 1.0, inline = False)
  b = foo_np(a)
  return B, b

def test2(A, a): # test compute with function inlined
  B = hcl.compute(A.shape, [A], lambda x, y: foo(A, x, y))
  b = foo_np(a)
  return B, b

def test3(A, a): # test compute without inling function
  B = hcl.compute(A.shape, [A], lambda x, y: foo(A, x, y), inline = False, extern_funcs = [foo])
  b = foo_np(a)
  return B, b

def test4(A, a): # test more complicated function
  B = hcl.compute(A.shape, [A], lambda x, y: foo2(A, x, y), inline = False, extern_funcs = [foo2])
  b = foo2_np(a)
  return B, b

def test5(A, a):
  B = hcl.compute((1,), [A], lambda x: A[2][3], inline = False)
  b = numpy.array([a[2][3]])
  return B, b

tests = [test5]

A = hcl.placeholder((10, 10), name = "A")

_a = numpy.random.rand(10, 10).astype("float32")

for test in tests:
  print "Testing...."
  B, _b = test(A, _a)

  s = tvm.create_schedule(B.op)
  func = tvm.build(s, [A, B])

  print tvm.lower(s, [A, B], simple_mode = True)

  # EXECUTION
  target = 'llvm'
  ctx = tvm.context(target, 0)
  a = tvm.nd.array(_a, ctx)
  b = tvm.nd.array(numpy.zeros((10, 10), dtype="float32"), ctx)
  if test == test5:
    b = tvm.nd.array(numpy.zeros((1,), dtype="float32"), ctx)

  func(a, b)

  b = b.asnumpy()
  numpy.testing.assert_allclose(b, _b, rtol=1e-5)


def foo3(a, B, x):
  return B[x] + a

a = hcl.var(name = "a", dtype = "float32")
B = hcl.placeholder((10,), name = "B", dtype = "float32")
C = hcl.compute(B.shape, [a, B], lambda x: foo3(a, B, x), name = "C", inline = False, extern_funcs = [foo3])

_a = 1.0
_B = numpy.random.rand(10).astype("float32")
_C = numpy.zeros(10, dtype="float32")

s = tvm.create_schedule(C.op)
func = tvm.build(s, [a, B, C])

ctx = tvm.context('llvm', 0)
t_a = _a
t_B = tvm.nd.array(_B)
t_C = tvm.nd.array(_C)

func(t_a, t_B, t_C)

numpy.testing.assert_allclose(t_C.asnumpy(), _B + 1.0, rtol=1e-5)
