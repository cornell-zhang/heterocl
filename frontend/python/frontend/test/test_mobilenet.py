from test_wrapper import *
from frontend import relay_parser
import sys
import keras
sys.trackbacklimit = 0

data_path = "/home/pbc48/install/datasets/imagenet_numpy/images/val/"
x_test = np.load(data_path+"x_test.npy")
y_test = np.load(data_path+"y_test.npy")
print(x_test.shape)
print(y_test.shape)
keras_model = keras.applications.MobileNet(include_top=True, weights='imagenet',
        input_shape=(224, 224, 3), classes=1000)
x = x_test/255
#x = x_train[0:49984,0:32,0:32,0:3]/255#.reshape(-1, 32,32,32,3)#.transpose(0,1,4,2,3)
x=np.reshape(x,(-1,25,224,224,3))
x=np.transpose(x,[0,1,4,2,3])
x= np.reshape(x,(-1,25,3,224,224))
y = y_test.reshape(-1,25)
test = 0
if(test == 0):
    test_wrapper(keras_model, "relay", "keras", "mobilenet",
                 x, y, (25, 1000), {'input_1': (25,3,224,224)}, batch_size=25)
elif(test == 1):
    f, params = relay_parser.get_relay_model("cifar10.h5", {"input_1":(32, 3,32,32)})
    for i in range(1562):
        _in = np.transpose(x_train[i],[0,3,1,2])
        _in = hcl.asarray(_in.reshape(32,3,32,32))
        _out = hcl.asarray(np.zeros((32, 10)))
        f(_in, *params, _out)
        print(np.argmax(_out.asnumpy(), axis=1), y_train[i])
