from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Reshape, ReLU
from keras.datasets import mnist
import numpy as np
import h5py

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_tr = x_train.reshape(60000, 784)
y_tr_b = to_categorical(y_train)
x_te = x_test.reshape(10000, 784)
y_te = to_categorical(y_test)
a = Input(shape=(784,))
b = Dense(200, activation='tanh')(a)
c = Dense(100)(b)
d = ReLU()(c)
e = Dense(50, activation='relu')(d)
f = Dense(10, activation='softmax')(e)
model = Model(inputs=a, outputs=f)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model, iterating on the data in batches of 32 samples
model.fit(x_tr, y_tr_b, epochs=25, batch_size=60)
y = model.evaluate(x_te, y_te)
y_t = model.predict(x_te)
print(y)
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
yaml_file.close()
np.save('x_mnist.npy', x_train)
np.save('y_mnist.npy', y_train)
model.save('model.hdf5')
