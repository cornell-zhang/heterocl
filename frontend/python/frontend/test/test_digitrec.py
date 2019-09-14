from test_wrapper import *
from keras.datasets import mnist
import sys
sys.trackbacklimit = 0
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 10, 784)
y_train = y_train.reshape(-1, 10)
print(x_train.shape)
print(y_train.shape)
test = 1
if(test == 0):
    test_wrapper("model.hdf5", "nnvm", "keras", "digitrec",
                 x_train, y_train, (10, 10), (784,), 10)
elif(test == 1):
    test_wrapper("model.hdf5", "relay", "keras", "digitrec",
                 x_train, y_train, (10, 10), {'input_1': (10, 784)})
elif(test == 2):
    f, params = from_nnvm("model.hdf5", 'keras', 'digitrec', (784,), 10)
    print(type(f))
    print(type(params[0]))
    for i in range(6000):
        _in = hcl.asarray(x_train[i].reshape(10, 784))
        _out = hcl.asarray(np.zeros((10, 10)))
        f(_in, *params, _out)
        # print(np.argmax(_out.asnumpy(),axis=1),y_train[i])
elif(test == 3):
    digitrec, params = get_relay_model("model.hdf5", (10, 784))
    for i in range(6000):
        _in = hcl.asarray(x_train[i].reshape(10, 784))
        _out = hcl.asarray(np.zeros((10, 10)))
        digitrec(_in, *params, _out)
        print(np.argmax(_out.asnumpy(), axis=1), y_train[i])
