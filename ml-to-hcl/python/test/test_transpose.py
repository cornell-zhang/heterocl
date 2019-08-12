import heterocl as hcl
import numpy as np
import hlib

hcl.init()
I = hcl.placeholder((4,3,2,1),"I")
def test(I,axes=[0,2,3,1]):
  return hlib.nn.transpose(I,axes)
s = hcl.create_schedule([I],test)
f = hcl.build(s)

data = np.random.randint(50,size=(4,3,2,1))
_out = hcl.asarray(np.zeros((4,2,1,3)))
data = hcl.asarray(data)
f(data,_out)
print(data.asnumpy())
print(_out.asnumpy())

I = hcl.placeholder((2,2,2,2,2,2),"I")
def test(I,axes=[0,2,3,1,5,4]):
  return hlib.nn.transpose(I,axes)
s = hcl.create_schedule([I],test)
f = hcl.build(s)

data = np.random.randint(50,size=(2,2,2,2,2,2))
_out = hcl.asarray(np.zeros((2,2,2,2,2,2)))
data = hcl.asarray(data)
f(data,_out)
print(data.asnumpy().shape)
print(_out.asnumpy().shape)

I = hcl.placeholder((3,3,3),"I")
def test(I,axes=[0,2,1]):
  return hlib.nn.transpose(I,axes)
s = hcl.create_schedule([I],test)
f = hcl.build(s)

data = np.random.randint(50,size=(3,3,3))
_out = hcl.asarray(np.zeros((3,3,3)))
data = hcl.asarray(data)
f(data,_out)
print(data.asnumpy().shape)
print(_out.asnumpy().shape)
