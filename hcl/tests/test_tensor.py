import hcl
import tvm
import numpy

def myfun(A, x, y):
  summ = 0
  for i in range(0, 5):
    summ = summ + A[x][i]
  return summ

A = hcl.placeholder((10, 10), name = "A")
B = hcl.compute((10, 10), [A], lambda x, y: myfun(A, x, y), inline = False, extern_funcs = [myfun])
#B = tvm.compute((10, 10), lambda x, y: A[x][y] + 1)

# EXECUTION
target = 'llvm'
ctx = tvm.context(target, 0)

print tvm.module.enabled('llvm')

s = tvm.create_schedule(B.op)
print tvm.lower(s, [A, B], simple_mode = True)
func = tvm.build(s, [A, B])

a = tvm.nd.array(numpy.random.rand(10, 10).astype("float32"), ctx)
b = tvm.nd.array(numpy.zeros((10, 10), dtype="float32"), ctx)

print a

func(a, b)

print b

