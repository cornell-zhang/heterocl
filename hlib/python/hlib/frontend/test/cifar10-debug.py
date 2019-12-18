from test_wrapper import *
from keras.datasets import cifar10
from keras.models import load_model, Model
from frontend import relay_parser
import heterocl as hcl
import hlib
import sys
import scipy
import numpy.testing as tst
hcl.init(hcl.Float())
model = load_model('cifar10.h5')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[0:49984,0:32,0:32,0:3]#.reshape(-1, 32,32,32,3)#.transpose(0,1,4,2,3)
#x_train.transpose([0,3,1,2])
print(x_train.dtype)
x_train=np.reshape(x_train,(-1,32,32,32,3))
y_train = y_train[0:49984].reshape(-1,32)

input_1 = hcl.placeholder((32,3,32,32))
param_1 = hcl.placeholder((32,3,3,3))
param_2 = hcl.placeholder((32,))
param_3 = hcl.placeholder((32,32,3,3))
param_4 = hcl.placeholder((32,))
param_5 = hcl.placeholder((64,32,3,3))
param_6 = hcl.placeholder((64,))
param_7 = hcl.placeholder((64,64,3,3))
param_8 = hcl.placeholder((64,))
param_9 = hcl.placeholder((512,2304))
param_10 = hcl.placeholder((512,))
param_11 = hcl.placeholder((10,512))
param_12 = hcl.placeholder((10,))
def cifar10_debug(input_1,
v_param_1, 
v_param_2,
v_param_3,
v_param_4,
v_param_5,
v_param_6,
v_param_7,
v_param_8,
v_param_9,
v_param_10,
v_param_11,
v_param_12):
    p0 = hlib.nn.conv2d(input_1,v_param_1,padding=[1,1],channels=32,kernel_size=[3,3])
    p1 = hlib.nn.bias_add(p0,v_param_2)
    p2 = hlib.nn.relu(p1)
    p3 = hlib.nn.conv2d(p2,v_param_3,channels=32,kernel_size=[3,3])
    p4 = hlib.nn.bias_add(p3,v_param_4)
    p5 = hlib.nn.relu(p4)
    p6 = hlib.nn.max_pool2d(p5,pool_size=[2,2], strides=[2,2])
    p7 = hlib.nn.conv2d(p6,v_param_5,padding=[1,1],channels=64,kernel_size=[3,3])
    p8 = hlib.nn.bias_add(p7,v_param_6)
    p9 = hlib.nn.relu(p8)
    p10 = hlib.nn.conv2d(p9,v_param_7,channels=64,kernel_size=[3,3])
    p11 = hlib.nn.bias_add(p10,v_param_8)
    p12 = hlib.nn.relu(p11)
    p13 = hlib.nn.max_pool2d(p12,pool_size=[2,2], strides=[2,2])
    p14 = hlib.nn.transpose(p13,axes=[0,2,3,1])
    p15 = hlib.nn.flatten(p14)
    p16 = hlib.nn.dense(p15,v_param_9,units=512)
    p17 = hlib.nn.bias_add(p16,v_param_10)
    p18 = hlib.nn.relu(p17)
    p19 = hlib.nn.dense(p18,v_param_11,units=10)
    p20 = hlib.nn.bias_add(p19,v_param_12)
    return hlib.nn.softmax(p20,axis=1)

f, params = relay_parser.get_relay_model("cifar10.h5", {"input_1":(32, 3,32,32)})
_in = np.transpose(x_train[0]/255,[0,3,1,2])
_in = _in.reshape(32,3,32,32)
input1 = hcl.asarray(_in)
s=hcl.create_schedule([input_1,param_1,param_2,param_3,param_4,
param_5,param_6,param_7,param_8,param_9,param_10,param_11,
param_12],cifar10_debug)
for par in params:
    par = hcl.asarray(par)
f = hcl.build(s)
out = hcl.placeholder((32,10))
out = hcl.asarray(np.zeros((32,10)))
out_layer_name = "dense_2"
#print(model.get_layer('activation_1'))
intermediate_output = Model(inputs=model.input,
outputs=model.get_layer(out_layer_name).output)
k_out =model.predict(x_train[0]/255)
#print(model.predict(x_train[0]))
f(input1,*params,out)
shape = out.shape
h_out = out.asnumpy()
#h_out = np.reshape(h_out,(shape[0],shape[3],shape[1],shape[2]))
#h_out = np.transpose(h_out,[0,2,3,1])
tst.assert_almost_equal(h_out,k_out,10**-6)
model.summary()
#print(scipy.special.softmax(output))