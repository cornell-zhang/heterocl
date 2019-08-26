import heterocl as hcl
import numpy as np
import hlib

hcl.init()
#A = hcl.placeholder((1,4,4,1),"A")
#def flat(A):
#  return hlib.nn.flatten(A)
#s = hcl.create_schedule([A],flat)
#f = hcl.build(s)
def flatten_test(in_size):
    def _flat_shape(in_size):
        length = 1
        for i in range(len(in_size)):
            length *= in_size[i]
        return (1, length)
    input1=hcl.placeholder(in_size)
    s = hcl.create_schedule([input1],hlib.nn.flatten)
    f = hcl.build(s)
    data = hcl.asarray(np.random.randint(50,size=in_size))
    _out = hcl.asarray(np.zeros(_flat_shape(in_size)))
    f(data,_out)
    return data.asnumpy(),_out.asnumpy()

def assert_flatten(data,out):
    assert(np.reshape(data,(1,-1)).shape==out.shape)
    assert(np.array_equal(np.reshape(data,(1,-1)),out))

#data = np.random.randint(50,size=(1,4,4,1))
#_out = hcl.asarray(np.zeros((1,16)))
#data = hcl.asarray(data)
#f(data,_out)
#print(data.asnumpy().reshape(4,4))
#print(_out.asnumpy().reshape(16))
assert_flatten(*flatten_test((1,4,4,1)))
assert_flatten(*flatten_test((16,4,4)))
assert_flatten(*flatten_test((4,)))

