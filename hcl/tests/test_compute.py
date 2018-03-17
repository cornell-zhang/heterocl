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

tests = [test0, test1, test2, test3, test4]

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

  func(a, b)

  b = b.asnumpy()
  numpy.testing.assert_allclose(b, _b, rtol=1e-5)

