import tvm
import hcl.util as _util

def myadd(A, B, C):
  i = tvm.var("i")
  return tvm.make.For(i, 0, 5, 0, 0,
      tvm.make.Store(C.data,
        tvm.make.Load("int32", A.data, i, _util.true), i, _util.true))

A = tvm.placeholder((5,), name="A")
B = tvm.placeholder((5,), name="B")
C = tvm.extern(A.shape, [A, B], lambda ins, outs: myadd(ins[0], ins[1], outs[0]), name="C")

s = tvm.create_schedule(C.op)

print tvm.lower(s, [A,B,C], simple_mode=True)

