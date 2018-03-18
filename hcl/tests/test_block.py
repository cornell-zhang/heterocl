import hcl
import tvm
import numpy

def foo(A, B):
  for i in range(0, 10):
    for j in range(0, 10):
      B[i][j] = A[i][j] + 1.0

def foo2(A, B):
  C = tvm.compute((10, 10), lambda x, y: 1.0, name = "C")
  for i in range(0, 10):
    for j in range(0, 10):
      B[i][j] = A[i][j] + C[i][j]

def foo_np(a):
  func = numpy.vectorize(lambda x: x + 1)
  return func(a)

def test0(A, B, a): # test basic hcl.compute
  C = hcl.block(foo, [A, B])
  b = foo_np(a)
  return C, b

def test1(A, B, a): # test basic hcl.compute
  C = hcl.block(foo2, [A, B])
  b = foo_np(a)
  return C, b

tests = [test1]

A = hcl.placeholder((10, 10), name = "A")
B = hcl.placeholder((10, 10), name = "B")

_a = numpy.random.rand(10, 10).astype("float32")

for test in tests:
  print "Testing...."
  C, _b = test(A, B, _a)

  s = tvm.create_schedule(C.op)
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

