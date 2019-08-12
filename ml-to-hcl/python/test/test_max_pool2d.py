import heterocl as hcl
import numpy as np
import hlib

hcl.init()
A = hcl.placeholder((1,16,16,1),"A")
stride = [1,1]
pooling = [2,2]
def maxpool(A,pooling=[2,2],stride=[1,1]):
  return hlib.nn.max_pool2d(A,pooling,stride,[0,0],'NHWC')
s = hcl.create_schedule([A],maxpool)
f = hcl.build(s)

data = np.random.randint(50,size=(1,16,16,1))
_out = hcl.asarray(np.zeros((1,15,15,1)))
data = hcl.asarray(data)
stride = hcl.asarray(stride)
pooling= hcl.asarray(pooling)
f(data,_out)
print(data.asnumpy().reshape(16,16))
print(_out.asnumpy().reshape(15,15))

A = hcl.placeholder((1,16,16,1),"A")
stride = [2,2]
pooling = [4,4]
def maxpool(A,pooling=[2,2],stride=[2,2]):
  return hlib.nn.max_pool2d(A,pooling,stride,[0,0],'NHWC')
s = hcl.create_schedule([A],maxpool)
f = hcl.build(s)

data = np.random.randint(50,size=(1,16,16,1))
_out = hcl.asarray(np.zeros((1,8,8,1)))
data = hcl.asarray(data)
stride = hcl.asarray(stride)
pooling= hcl.asarray(pooling)
f(data,_out)
print(data.asnumpy().reshape(16,16))
print(_out.asnumpy().reshape(8,8))


