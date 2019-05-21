import heterocl as hcl
import numpy as np
import hlib

hcl.init()
A = hcl.placeholder((1,4,4,1),"A")
def flat(A):
  return hlib.nn.flatten(A)
s = hcl.create_schedule([A],flat)
f = hcl.build(s)

data = np.random.randint(50,size=(1,4,4,1))
_out = hcl.asarray(np.zeros((1,16)))
data = hcl.asarray(data)
f(data,_out)
print(data.asnumpy().reshape(4,4))
print(_out.asnumpy().reshape(16))


