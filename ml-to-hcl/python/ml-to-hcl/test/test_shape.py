import heterocl as hcl
import heterocl.tvm as tvm
import numpy as np
import hlib

def expand_dim_test(in_shape,axis,new_axis):
    input1 = hcl.placeholder(in_shape)
    def func(input1,axis=axis,new_axis=new_axis):
        return hlib.nn.expand_dims(input1,axis,new_axis)
    s = hcl.create_schedule([input1],func)
    f = hcl.build(s)
    _in = np.random.randint(50,size=in_shape)
    real_out = _in
    for i in range(new_axis):
        real_out  = np.expand_dims(real_out,axis)
    def _new_shape(in_shape,axis,new_axis):
        new_shape = []
        for i in range(axis):
            new_shape.append(in_shape[i])
        for i in range(new_axis):
            new_shape.append(1)
        for i in range(len(in_shape)-axis):
            new_shape.append(in_shape[i+axis])
        return new_shape
    _out = hcl.asarray(np.zeros(_new_shape(in_shape,axis,new_axis)))
    _in = hcl.asarray(_in)
    f(_in,_out)
    return _in.asnumpy(),_out.asnumpy(),real_out

def squeeze_test(in_shape,axis=None):
    input1 = hcl.placeholder(in_shape)
    def func(input1,axis=axis):
        return hlib.nn.squeeze(input1,axis)
    s = hcl.create_schedule([input1],func)
    f = hcl.build(s)
    _in = np.random.randint(50,size=in_shape)
    real_out = _in
    real_out  = np.squeeze(real_out,axis)
    def _new_shape(in_shape,axis):
        new_shape = []
        if(axis==None):
            for i in range(len(in_shape)):
                if in_shape[i] != 1:
                    new_shape.append(in_shape[i])
        else:
            for i in range(len(in_shape)):
                if not i in axis:
                    new_shape.append(in_shape[i])
        return new_shape
    _out = hcl.asarray(np.zeros(_new_shape(in_shape,axis)))
    _in = hcl.asarray(_in)
    f(_in,_out)
    return _in.asnumpy(),_out.asnumpy(),real_out

def assert_expand_dim(_in,real_out,out):
    assert(np.array_equal(real_out.shape,out.shape))

def assert_squeeze(_in,real_out,out):
    assert(np.array_equal(real_out.shape,out.shape))

assert_expand_dim(*expand_dim_test((3,3),1,1))
assert_expand_dim(*expand_dim_test((3,3,3,3,3,3),2,2))

assert_squeeze(*squeeze_test((1,1,3,3,3,3)))
assert_squeeze(*squeeze_test((1,1,3,3,3,3),axis=(1,)))
assert_squeeze(*squeeze_test((1,1,3,3,3,3),axis=(0,)))
assert_squeeze(*squeeze_test((1,1,3,3,3,3),axis=(1,0)))

