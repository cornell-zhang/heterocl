import keras
import numpy as np
import heterocl as hcl
import tvm
import tvm.relay as relay
import tvm.relay.frontend as relay_front
import numpy.testing as tst
from frontend.relay_parser import relay_parser, get_relay_model

def verify_keras_frontend(keras_model):
    assert(keras.backend.backend() == 'tensorflow')
    if(keras_model==None):
        return
    in_shapes = []
    for layer in keras_model._input_layers:
        in_shapes.append(tuple(dim.value if dim.value is not None else 1 for dim in layer.input.shape))

    def get_keras_output(xs, dtype='float32'):
        return keras_model.predict(xs)

    def get_hcl_output(xs, dtype='float32'):
        shape_dict = {name: x.shape for (name, x) in zip(keras_model.input_names, xs)}
        #return relay_front.from_keras(keras_model, shape_dict)
        print("SD:",shape_dict)
        return get_relay_model(keras_model, shape_dict, 'keras')

    xs = [np.random.uniform(size=shape, low=1, high=10) for shape in in_shapes]
    keras_out = get_keras_output(xs)
    print(len(keras_out))
    #return get_hcl_output(xs)
    f,params = get_hcl_output(xs)
    out = []
    print("here")
    if(isinstance(keras_out,tuple)):
        for k_out in keras_out:
            out.append(hcl.asarray(np.zeros(k_out.shape)))
    else:
        out.append(hcl.asarray(np.zeros(keras_out.shape)))
    for i in range(len(xs)):
        xs[i] = hcl.asarray(xs[i])
    print("down here")
    f(*xs,*params,*out)
    print("below function")
    if(isinstance(keras_out,tuple)):
        for i in range(len(keras_out)):
            tst.assert_almost_equal(out[i].asnumpy(),keras_out[i],10**-6)
    else:
        tst.assert_almost_equal(out[0].asnumpy(),keras_out,10**-6)
def merge_test(shape):
    x = keras.layers.Input(shape=shape)
    y = keras.layers.Input(shape=shape)
    z = keras.layers.Input(shape=shape)
    merge_funcs = [keras.layers.Add(),
                   #keras.layers.Subtract(),
                   keras.layers.Multiply(),
                   keras.layers.Maximum()]#,
                   #keras.layers.Average(),
                   #keras.layers.Concatenate()]
    for merge_func in merge_funcs:
        if isinstance(merge_func, (keras.layers.merge.Subtract, keras.layers.merge.Dot)):
            out = merge_func([x, y])
        else:
            out = merge_func([x, y, z])
        keras_model = keras.models.Model([x,y,z], out)
        verify_keras_frontend(keras_model)   

def merge_2_test(shape):
    x = keras.layers.Input(shape=shape)
    y = keras.layers.Input(shape=shape)
    merge_funcs = [keras.layers.Subtract()]
                   #keras.layers.Average(),
    for merge_func in merge_funcs:
        out = merge_func([x, y])
        keras_model = keras.models.Model([x,y], out)
        verify_keras_frontend(keras_model)   

def merge_conv_test():
    data = keras.layers.Input(shape=(32,32,3))
    x = keras.layers.Conv2D(8, (3, 3), padding="valid")(data)
    y = keras.layers.Conv2D(8, (3, 3), padding="valid")(data)
    z = keras.layers.Conv2D(8, (3, 3), padding="valid")(data)
    merge_funcs = [keras.layers.Add(),
                   keras.layers.Subtract(),
                   keras.layers.Multiply(),
                   keras.layers.Maximum()]#,
                   #keras.layers.Average(),
    #               keras.layers.Concatenate()]
    for merge_func in merge_funcs:
        if isinstance(merge_func, (keras.layers.merge.Subtract, keras.layers.merge.Dot)):
            out = merge_func([x, y])
        else:
            out = merge_func([x, y, z])
    out = keras.layers.Add()([x,y])
    keras_model = keras.models.Model(data, out)
    verify_keras_frontend(keras_model)

def pooling_test(shape):
    data = keras.layers.Input(shape=shape)
    x = keras.layers.MaxPooling2D()(data)
    y = keras.layers.MaxPooling2D()(x)
    z = keras.layers.AveragePooling2D()(y)
    w = keras.layers.AveragePooling2D()(z)
    keras_model = keras.models.Model(data, w)
    verify_keras_frontend(keras_model) 

def merge_and_pool_test(shape):
    data = keras.layers.Input(shape=shape)
    x = keras.layers.MaxPooling2D()(data)
    z = keras.layers.MaxPooling2D()(x)
    y = keras.layers.AveragePooling2D()(data)
    w = keras.layers.AveragePooling2D()(y)
    out = keras.layers.Add()([z,w])
    keras_model = keras.models.Model(data, out)
    verify_keras_frontend(keras_model) 

def merge_out_tup_test(shape):
    data = keras.layers.Input(shape=shape)
    x = keras.layers.MaxPooling2D()(data)
    z = keras.layers.MaxPooling2D()(x)
    y = keras.layers.AveragePooling2D()(data)
    w = keras.layers.AveragePooling2D()(y)
    keras_model = keras.models.Model(data, [z,w])
    verify_keras_frontend(keras_model)

def merge_just_conv_test():
    data = keras.layers.Input(shape=(3,3,3))
    out = keras.layers.Conv2D(3, (3, 3), padding="same")(data)
    keras_model = keras.models.Model(data, out)
    verify_keras_frontend(keras_model)

def dot_test():
    data1 = keras.layers.Input(shape=(2, 2))
    data2 = keras.layers.Input(shape=(2, 2))
    merge_funcs = [keras.layers.Dot(axes=[1, 2]),
                   keras.layers.Dot(axes=[2, 1]),
                   keras.layers.Dot(axes=[1, 1]),
                   keras.layers.Dot(axes=[2, 2]),
                   keras.layers.Dot(axes=1),
                   keras.layers.Dot(axes=2)]
    for merge_func in merge_funcs:
        out = merge_func([data1, data2])
        keras_model = keras.models.Model([data1, data2], out)
        verify_keras_frontend(keras_model)

def sequential_test():
    keras_model = keras.models.Sequential([
        keras.layers.Dense(16, input_dim=32, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    verify_keras_frontend(keras_model)

def simple_pool_test():
    data = keras.layers.Input(shape=(3, 3, 1))
    # maxpool
    x = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)
    # avgpool
    y = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1,1), padding='valid')(data)
    keras_model = keras.models.Model(data, y)
    verify_keras_frontend(keras_model)

def reshape_test():
    # input_shape len is 3, target_shape len is 3
    data = keras.layers.Input(shape=(32, 32, 3))
    x = keras.layers.Reshape(target_shape=(16, 64, 3))(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)
    # input_shape len is 3, target_shape len is 2
    data = keras.layers.Input(shape=(32, 8, 3))
    x = keras.layers.Reshape(target_shape=(256, 3))(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)
    # input_shape len is 2, target_shape len is 3
    data = keras.layers.Input(shape=(256, 3))
    x = keras.layers.Reshape(target_shape=(8, 32, 3))(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)
    # input_shape len is 2, target_shape len is 1
    data = keras.layers.Input(shape=(2, 8))
    x = keras.layers.Reshape(target_shape=(16,))(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)
    # input_shape len is 1, target_shape len is 2
    data = keras.layers.Input(shape=(16,))
    x = keras.layers.Reshape(target_shape=(4, 4))(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)
    # input_shape len is 2, target_shape len is 2
    data = keras.layers.Input(shape=(2, 8))
    x = keras.layers.Reshape(target_shape=(4, 4))(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model)

def rnn_test():
    data = keras.layers.Input(shape=(1, 32))
    rnn_funcs = [keras.layers.LSTM(units=16, return_state=False,
                    recurrent_activation='sigmoid', activation='tanh'),
                 keras.layers.SimpleRNN(units=16, return_state=False,
                    activation='tanh'),
                 keras.layers.GRU(units=16, return_state=False,
                    recurrent_activation='sigmoid', activation='tanh')]
    for rnn_func in rnn_funcs:
        x = rnn_func(data)
        keras_model = keras.models.Model(data, x)
        verify_keras_frontend(keras_model)

#merge_test((3,3))
#merge_test((10,7,4))
#merge_2_test((3,3))
#pooling_test((3,3,1))
#pooling_test((32,32,16))
#pooling_test((32,16,32))
#pooling_test((16,32,32))
#dot_test()
#sequential_test()
#rnn_test()
#reshape_test()
#simple_pool_test()
#merge_and_pool_test((32,32,32))
#merge_out_tup_test((32,32,32))
#merge_just_conv_test()
merge_conv_test()
print("All Passed!")