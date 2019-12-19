import heterocl as hcl
import hlib
import numpy as np

hcl.init()

def bias_add_test(d_shape,b_shape,axis=1):
    data=hcl.placeholder(d_shape)
    bias=hcl.placeholder(b_shape)
    def func(data, bias, axis=axis):
        return hlib.op.nn.bias_add(data, bias, axis=axis)
    s = hcl.create_schedule([data,bias], func)
    f = hcl.build(s)
    _in = np.random.randint(3,size=d_shape)
    b = np.random.randint(3,size=b_shape)
    out = hcl.asarray(np.zeros(d_shape))
    f(hcl.asarray(_in),hcl.asarray(b),out)
    return out.asnumpy(),_in,b

print(bias_add_test((3,3,3),(3,),1))
print(bias_add_test((3,3,3),(3,),0))
print(bias_add_test((3,3,3),(3,),2))
print(bias_add_test((3,3,3,3),(3,),0))
print(bias_add_test((3,3,3,3),(3,),1))
print(bias_add_test((3,3,3,3),(3,),2))
print(bias_add_test((3,3,3,3),(3,),-1))
