import heterocl as hcl
import hlib
import numpy as np

hcl.init()
data = hcl.placeholder((10,10),"data")
bias = hcl.placeholder((10,),"bias")

def bias_add(data,bias,axis=-1):
    return hlib.nn.bias_add(data,bias,axis)

#Basic test

data = hcl.placeholder((10,10),"data")
bias = hcl.placeholder((10,),"bias")

sch_1 = hcl.create_schedule([data,bias],bias_add)
test_1 = hcl.build(sch_1)

_data = np.random.randint(20,size=(10,10))
_bias = np.random.randint(20,size=(10,))
_out  = np.zeros((10,10))
_data = hcl.asarray(_data)
_bias = hcl.asarray(_bias)
_out  = hcl.asarray(_out)
test_1(_data,_bias,_out)
print(_data)
print(_bias)
print(_out)

#bigger,shaped axis test

data = hcl.placeholder((5,5,5),"data")
bias = hcl.placeholder((5,),"bias")

sch_2 = hcl.create_schedule([data,bias],bias_add)
test_2 = hcl.build(sch_2)

_data = np.random.randint(20,size=(5,5,5))
_bias = np.random.randint(20,size=(5,))
_out  = np.zeros((5,5,5))
_data = hcl.asarray(_data)
_bias = hcl.asarray(_bias)
_out  = hcl.asarray(_out)
test_2(_data,_bias,_out)
print(_data)
print(_bias)
print(_out)

#multi-dimensional bias
data = hcl.placeholder((5,5,5),"data")
bias = hcl.placeholder((5,5,),"bias")

sch_3 = hcl.create_schedule([data,bias],bias_add)
test_3 = hcl.build(sch_3)

_data = np.random.randint(20,size=(5,5,5))
_bias = np.random.randint(20,size=(5,5,))
_out  = np.zeros((5,5,5))
_data = hcl.asarray(_data)
_bias = hcl.asarray(_bias)
_out  = hcl.asarray(_out)
test_3(_data,_bias,_out)
print(_data)
print(_bias)
print(_out)
