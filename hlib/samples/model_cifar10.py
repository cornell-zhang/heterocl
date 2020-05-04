from .test_wrapper import *
from keras.datasets import cifar10
from keras.models import load_model
from hlib.frontend import relay_parser
import sys
sys.trackbacklimit = 0

batch = 1
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x = x_test[0:9984, 0:32, 0:32, 0:3] / 255
x_train_keras = np.reshape(x, (-1, batch, 32, 32, 3))
x = np.transpose(x_train_keras, [0, 1, 4, 2, 3])
x = np.reshape(x, (-1, batch, 3, 32, 32))
y = y_test[0:9984].reshape(-1, batch)
test = 1
if(test == 0):
    test_wrapper("cifar10.h5", "relay", "keras", "cifar10",
                 x, y, (batch, 10), {'input_1': (batch, 3, 32, 32)}, batch_size=32)
elif(test == 1):
    keras_model = load_model('cifar10.h5')
    correct = 0
    total = 0
    for i in range(y.shape[0]):
        x_p = keras_model.predict(x_train_keras[i])
        if(np.argmax(x_p, axis=1) == y[i]):
            correct += 1
        total += 1
    print("accuracy:", correct / (total * 1.0))
