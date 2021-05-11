
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json

def scale(im, nR, nC):
  nR0 = len(im)     # source number of rows 
  nC0 = len(im[0])  # source number of columns 
  return [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]
             for c in range(nC)] for r in range(nR)]

# Model / data parameters
num_classes = 10
input_shape = (1,30,30)
size = 30

def train_model():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    scale_x_train = np.zeros(shape=(x_train.shape[0], size, size))
    scale_x_test = np.zeros(shape=(x_test.shape[0], size, size))
    for num in range(x_train.shape[0]):
        new_img = scale(x_train[num], size, size)
        scale_x_train[num] = new_img
    for num in range(x_test.shape[0]):
        new_img = scale(x_test[num], size, size)
        scale_x_test[num] = new_img

    # Scale images to the [0, 1] range
    x_train = scale_x_train.astype("float32") / 255
    x_test = scale_x_test.astype("float32") / 255

    # Make sure images have shape (1,28,28)
    # x_train = np.expand_dims(x_train, axis=0)
    # x_test = np.expand_dims(x_test, axis=0)
    x_train = x_train.reshape((60000,1,size,size))
    x_test = x_test.reshape((10000,1,size,size))
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # convnet example 
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(16, kernel_size=(3,3), use_bias=False, data_format="channels_first"),
            layers.Conv2D(64, kernel_size=(3,3), activation="relu", use_bias=False, data_format="channels_first"),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax", use_bias=False),
        ]
    )

    model.summary()
    batch_size = 128
    epochs = 15

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Save the model to files: https://stackoverflow.com/a/44653102
    # First conv layer and dense layer
    with open('convnet.npy', 'wb') as f:
        np.save(f, model.layers[0].get_weights())
        np.save(f, model.layers[1].get_weights())
        np.save(f, model.layers[4].get_weights())

    with open('input_data.npy', 'wb') as fp:
        np.save(fp, x_test)
        np.save(fp, y_test)
    
    # Save model into h5
    model_json = model.to_json()
    with open("model_convnet.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_convnet_weight.h5")

def main():
    if not os.path.exists('convnet.npy'):
        train_model()

    # Loading model and get intermediate layer output
    json_file = open('model_convnet.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model_convnet_weight.h5")

    # test intermediate output of first 1st data
    layer_names = ['conv2d', 'conv2d_1', 'flatten', 'dense']
    with open('input_data.npy', 'rb') as fp:
        test_data = np.load(fp)
        data = test_data[0]
        data = data[np.newaxis, :]

    with open('intermediate.npy', 'wb') as fp:
        for layer_name in layer_names:
            intermediate_layer_model = keras.Model(inputs=loaded_model.input,
                                             outputs=loaded_model.get_layer(layer_name).output)       
            intermediate_output = intermediate_layer_model.predict(data)
            np.save(fp, intermediate_output)

if __name__ == "__main__":
    main()

