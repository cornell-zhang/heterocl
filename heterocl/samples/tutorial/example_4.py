import heterocl as hcl
import numpy as np

hcl.config.init_dtype = hcl.UInt() # set the max precision
shape = (20,)

def popcount(n):
  l = hcl.local(0)
  with hcl.for_(0, 32) as i:
    l[0] += n[i]
  return l[0]

def top(dtype):
  A = hcl.placeholder(shape)
  B = hcl.compute(shape, [A], lambda x: popcount(A[x]))

  hcl.downsize(B, dtype) # apply different quantization schemes

  s = hcl.create_schedule(B)
  return hcl.build(s, [A, B])

nA = np.random.randint(1 << 32, size = shape)
nB = np.zeros(shape)

print nA

# 5-bit should be enough for B since the maximum popcount result is 31
for i in range(1, 8):
  dtype = hcl.UInt(i)
  _A = hcl.asarray(nA, hcl.UInt())
  _B = hcl.asarray(nB, dtype)
  top(dtype)(_A, _B)
  print str(dtype) + ": " + str(_B)
