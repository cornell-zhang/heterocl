import hcl
import tvm
import numpy

def foo(A, x, y):
  return A[x][y] + 1.0

def foo_np(a):
  func = numpy.vectorize(lambda x: x + 1)
  return func(a)

def test0(A, a): # test basic hcl.compute
  B = hcl.update(A, [], lambda x, y: A[x][y] + 1.0)
  b = foo_np(a)
  return B, b

def test1(A, a): # test compute with function inlined
  B = hcl.update(A, [], lambda x, y: foo(A, x, y), extern_funcs = [foo])
  b = foo_np(a)
  return B, b


tests = [test0, test1]

A = hcl.placeholder((10, 10), name = "A")

_a = numpy.random.rand(10, 10).astype("float32")

for test in tests:
  print "Testing...."
  B, _b = test(A, _a)

  s = tvm.create_schedule(B.op)
  func = tvm.build(s, [A])

  print tvm.lower(s, [A], simple_mode = True)

  # EXECUTION
  target = 'llvm'
  ctx = tvm.context(target, 0)
  a = tvm.nd.array(_a, ctx)

  func(a)

  a = a.asnumpy()
  numpy.testing.assert_allclose(a, _b, rtol=1e-5)

