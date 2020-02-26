from .test_wrapper import *
from keras.datasets import mnist
from keras.models import load_model
import sys
sys.trackbacklimit = 0
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = x_train.reshape(-1, 1, 784)
y = y_train.reshape(-1, 1)
test = 0
if test == 0:
    test_wrapper("model.hdf5", "relay", "keras", "digitrec",
                 x, y, (1, 10), {'input_1': (1, 784)})
elif(test == 1):
    keras_model = load_model('model.hdf5')
    correct = 0
    total = 0
    for i in range(y.shape[0]):
        x_p = keras_model.predict(x[i])
        if(np.argmax(x_p, axis=1) == y[i]):
            correct += 1
        total += 1
    print("accuracy:", correct / (total * 1.0))
