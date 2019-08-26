from collections import OrderedDict
import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np

dtype = hcl.Int()
hcl.init()

max = hcl.reducer(-10000, lambda x, y: tvm.make.Max(x, y), dtype)
min = hcl.reducer(10000, lambda x, y: tvm.make.Min(x, y), dtype)
sum = hcl.reducer(0, lambda x,y: x + y, dtype)
prod= hcl.reducer(1, lambda x,y: x * y, dtype)

#math functions

def exp(input1,name='exp'):
    return hcl.compute(input1.shape,lambda *x: hcl.exp(input1[x]),name=name)

def log(input1, name='log'):
    return hcl.compute(input1.shape,lambda *x: hcl.log(input1[x]),name=name)

def sqrt(input1, name='sqrt'):
    return hcl.compute(input1.shape,lambda *x: hcl.sqrt(input1[x]),name=name)

def sigmoid(input1, name='sigmoid'):
    return hcl.compute(input1.shape,lambda *x: hcl.sigmoid(input1[x]),name=name)

def tanh(x, name="tanh"):
    return hcl.compute(x.shape, lambda *args: hcl.tanh(x[args]), name,
                       attrs=OrderedDict([('app_name', tvm.make.StringImm('tanh'))]))

def max(data,axis):
    pass

#numpy_like functions

def full(in_shape=(1,),fill_val=1,dtype=dtype,name='full'):
    hcl.init(dtype)
    return hcl.compute(shape,lambda *x: hcl.cast(dtype,fill_val),name=name)
    

def full_like(array,fill_val,dtype=None,name='full_like'):
    if dtype==None:
        dtype=array.dtype
    hcl.init(dtype)
    return hcl.compute(array.shape,lambda *x: hcl.cast(dtype,fill_val),name=name)

def ones(in_shape,dtype=dtype,name='ones'):
    hcl.init(dtype)
    return hcl.compute(in_shape,lambda *x: hcl.cast(dtype,1),name=name)

def ones_like(array,dtype=None,name='ones_like'):
    if dtype==None:
        dtype=array.dtype
    hcl.init(dtype)
    return hcl.compute(array.shape,lambda *x: hcl.cast(dtype,1),name=name)

def zeros(in_shape,dtype=dtype,name='zeros'):
    hcl.init(dtype)
    return hcl.compute(in_shape,lambda *x: hcl.cast(dtype,0),name=name)

def zeros_like(array,dtype=None,name='zeros_like'):
    if dtype==None:
        dtype=array.dtype
    hcl.init(dtype)
    return hcl.compute(array.shape,lambda *x: hcl.cast(dtype,0),name=name)

