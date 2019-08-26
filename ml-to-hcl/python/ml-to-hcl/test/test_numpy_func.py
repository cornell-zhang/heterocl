import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np
import hlib

hcl.init()

def full_test(shape,fill_val=1,dtype=None):
    def full(in_shape=(1,),fill_val=1,dtype=dtype,name='full'):
        hcl.init(dtype)
        return hcl.compute(shape,lambda *x: hcl.cast(dtype,fill_val),name=name)
    def func(shape=shape,fill_val=fill_val,dtype=dtype):
        return full(shape,fill_val,dtype=dtype)
    s = hcl.create_schedule([],func)
    f = hcl.build(s)
    out = hcl.asarray(np.zeros(shape))
    f(out)
    return out.asnumpy()

def full_like_test(array_shape,fill_val=1,dtype=None):
    def full_like(array,fill_val,dtype=None,name='full_like'):
        if dtype==None:
            dtype=array.dtype
        hcl.init(dtype)
        return hcl.compute(array.shape,lambda *x: hcl.cast(dtype,fill_val),name=name)
    array = hcl.placeholder(array_shape)
    def func(array,fill_val=fill_val):
        return full_like(array,fill_val)
    s = hcl.create_schedule(array,func)
    f = hcl.build(s)
    out = hcl.asarray(np.zeros(array_shape))
    _array = hcl.asarray(np.zeros(array_shape))
    f(_array,out)
    return out.asnumpy()

print(full_test((3,3),fill_val=5.01,dtype=hcl.Float()))

print(full_like_test((3,3),fill_val=5.01,dtype=hcl.Float()))
