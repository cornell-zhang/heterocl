import tvm
import hcl.util as _util

def myadd(A, B, C):
  i = tvm.var("i")
  return tvm.make.For(i, 0, 5, 0, 0,
      tvm.make.Store(C.data,
        tvm.make.Load("int32", A.data, i, _util.true), i, _util.true))

def foo(A, x):
  b = 0
  for i in range(0, 3):
    b += A[x + i]
  return b

A = tvm.placeholder((5,), name="A")
B = tvm.placeholder((5,), name="B")
C = tvm.extern(A.shape, [A, B], lambda ins, outs: myadd(ins[0], ins[1], outs[0]), name="C")
D = tvm.compute(A.shape, lambda x: foo(A, x), name = "D")
E = tvm.compute(A.shape, lambda x: x)

s = tvm.create_schedule(C.op)

print D.op.body
#print s.stage_map
#print tvm.lower(s, [A, B, C], simple_mode=True)


