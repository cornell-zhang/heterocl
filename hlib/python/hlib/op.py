from collections import OrderedDict
import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np

dtype = hcl.Float()

max = hcl.reducer(-1, lambda x, y: tvm.make.Max(x, y), dtype)
min = hcl.reducer(2**(dtype.bits-1)-1, lambda x, y: tvm.make.Min(x, y), dtype)

#elemwise functions

def logical_and(input1,input2,name='logical_and'):
    return hcl.compute(input.shape,lambda *x: input1[x]&input2[x],name=name)

def logical_or(input1,input2,name='logical_or'):
    return hcl.compute(input.shape,lambda *x: input1[x]|input2[x],name=name)

def logical_not(input1,name='logical_not'):
    return hcl.compute(input.shape,lambda *x: not input1[x],name=name)

def elemwise_add(input1,input2,name='elemwise_add'):
    return hcl.compute(input.shape,lambda *x: input1[x]+input2[x],name=name)

def elemwise_sub(input1,input2,name='elemwise_sub'):
    return hcl.compute(input.shape,lambda *x: input1[x]-input2[x],name=name)

def elemwise_mul(input1,input2,name='elemwise_mul'):
    return hcl.compute(input.shape,lambda *x: input1[x]*input2[x],name=name)

def elemwise_div(input1,input2,name='elemwise_div'):
    return hcl.compute(input.shape,lambda *x: input1[x]/input2[x],name=name)

def elemwise_mod(input1,input2,name='elemwise_mod'):
    return hcl.compute(input.shape,lambda *x: input1[x]%input2[x],name=name)

def elemwise_pow(input1,input2,name='elemwise_pow'):
    return hcl.compute(input.shape,lambda *x: input1[x]**input2[x],name=name)


#broadcast functions

def _broadcast(shape,*indices):
    axes = []
    indices=indices[0]
    for i in range(len(shape)):
        if(shape[i]==1):
            axes.append(0)
        else:
            axes.append(indices[i])
    axes = tuple(axes)
    return axes

def broadcast_add(input1,input2,name='broadcast_add'):
    return hcl.compute(input1.shape,lambda *x: input1[x]+input2[_broadcast(input2.shape,x)],name=name)

def broadcast_sub(input1,input2,name='broadcast_sub'):
    return hcl.compute(input1.shape,lambda *x: input1[x]-input2[_broadcast(input2.shape,x)],name=name)

def broadcast_mul(input1,input2,name='broadcast_mul'):
    return hcl.compute(input1.shape,lambda *x: input1[x]*input2[_broadcast(input2.shape,x)],name=name)

def broadcast_div(input1,input2,name='broadcast_div'):
    return hcl.compute(input1.shape,lambda *x: input1[x]/input2[_broadcast(input2.shape,x)],name=name)

def broadcast_mod(input1,input2,name='broadcast_mod'):
    return hcl.compute(input1.shape,lambda *x: input1[x]%input2[_broadcast(input2.shape,x)],name=name)

def broadcast_pow(input1,input2,name='broadcast_pow'):
    return hcl.compute(input1.shape,lambda *x: tvm.power(input1[x],input2[_broadcast(input2.shape,x)]),name=name)

def broadcast_equal(input1,input2,name='broadcast_equal'):
    return hcl.compute(input1.shape, lambda *x: 1 if input1[x]==input2[_broadcast(input2.shape,x)] else 0,name=name)

def broadcast_not_equal(input1,input2,name='broadcast_not_equal'):
    return hcl.compute(input1.shape, lambda *x: 1 if input1[x]!=input2[_broadcast(input2.shape,x)] else 0,name=name)

def broadcast_greater(input1,input2,name='broadcast_greater'):
    return hcl.compute(input1.shape,lambda *x: input1[x]>input2[_broadcast(input2.shape,x)],name=name)

def broadcast_less(input1,input2,name='broadcast_less'):
    return hcl.compute(input1.shape,lambda *x: input1[x]<input2[_broadcast(input2.shape,x)],name=name)

def broadcast_greater_equal(input1,input2,name='broadcast_greater_equal'):
    return hcl.compute(input1.shape,lambda *x: input1[x]>=input2[_broadcast(input2.shape,x)],name=name)

def broadcast_less_equal(input1,input2,name='broadcast_less_equal'):
    return hcl.compute(input1.shape,lambda *x: input1[x]<=input2[_broadcast(input2.shape,x)],name=name)

def broadcast_right_shift(input1,input2,name='broadcast_right_shift'):
    return hcl.compute(input1.shape,lambda *x: input1[x]<<input2[_broadcast(input2.shape,x)],name=name)

def broadcast_left_shift(input1,input2,name='broadcast_left_shift'):
    return hcl.compute(input1.shape,lambda *x: input1[x]>>input2[_broadcast(input2.shape,x)],name=name)

def broadcast_max(input1,input2,name='broadcast_max'):
    return hcl.compute(input1.shape,lambda *x: max(input1[x],input2[_broadcast(input2.shape,x)]),name=name)

def broadcast_min(input1,input2,name='broadcast_min'):
    return hcl.compute(input1.shape,lambda *x: min(input1[x],input2[_broadcast(input2.shape,x)]),name=name)

def broadcast_and(input1,input2,name='broadcast_and'):
    return hcl.compute(input1.shape,lambda *x: input1[x]&input2[_broadcast(input2.shape,x)],name=name)

def broadcast_or(input1,input2,name='broadcast_or'):
    return hcl.compute(input1.shape,lambda *x: input1[x]|input2[_broadcast(input2.shape,x)],name=name)

def broadcast_xor(input1,input2,name='broadcast_xor'):
    return hcl.compute(input1.shape,lambda *x: input1[x]^input2[_broadcast(input2.shape,x)],name=name)


