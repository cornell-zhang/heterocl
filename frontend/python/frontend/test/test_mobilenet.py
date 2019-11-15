from test_wrapper import *
import tensorflow_datasets as tfds
from frontend import relay_parser
import sys
sys.trackbacklimit = 0

imagenet = tfds.load(name="imagenet2012", split=tfds.Split.VALIDATION)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_test.shape)
x = x_train[0:49984,0:32,0:32,0:3]/255
#x = x_train[0:49984,0:32,0:32,0:3]/255#.reshape(-1, 32,32,32,3)#.transpose(0,1,4,2,3)
x=np.reshape(x,(-1,32,32,32,3))
x=np.transpose(x,[0,1,4,2,3])
x= np.reshape(x,(-1,32,3,32,32))
y = y_train[0:49984].reshape(-1,32)
test = 0
if(test == 0):
    test_wrapper("cifar10.h5", "relay", "keras", "cifar10",
                 x, y, (32, 10), {'input_1': (32,3,32,32)}, batch_size=32)
elif(test == 1):
    f, params = relay_parser.get_relay_model("cifar10.h5", {"input_1":(32, 3,32,32)})
    for i in range(1562):
        _in = np.transpose(x_train[i],[0,3,1,2])
        _in = hcl.asarray(_in.reshape(32,3,32,32))
        _out = hcl.asarray(np.zeros((32, 10)))
        f(_in, *params, _out)
        print(np.argmax(_out.asnumpy(), axis=1), y_train[i])