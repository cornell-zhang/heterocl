import keras
import numpy as np
import heterocl as hcl
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

    xs = [np.random.uniform(size=shape, low=-50, high=50) for shape in in_shapes]
    keras_out = get_keras_output(xs)
    #return get_hcl_output(xs)
    f,params = get_hcl_output(xs)
    out = hcl.asarray(np.zeros(keras_out.shape))
    for i in range(len(xs)):
        xs[i] = hcl.asarray(xs[i])
    f(*xs,out)
    print("Here")
    tst.assert_almost_equal(out.asnumpy(),keras_out,10**-6)

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
    data = keras.layers.Input(shape=(32,3,32))
    x = keras.layers.Conv2D(8, (3, 3), padding="same")(data)
    y = keras.layers.Conv2D(8, (3, 3), padding="same")(x)
    #merge_funcs = [#keras.layers.Add(),
                   #keras.layers.Subtract(),
                   #keras.layers.Multiply(),
                   #keras.layers.Maximum(),
                   #keras.layers.Average(),
    #               keras.layers.Concatenate()]
    #for merge_func in merge_funcs:
    #    if isinstance(merge_func, (keras.layers.merge.Subtract, keras.layers.merge.Dot)):
    #        out = merge_func([x, y])
    #    else:
    #        out = merge_func([x, y, z])
    out = keras.layers.Add()([x,y])
    keras_model = keras.models.Model(data, out)
    return keras_model

def pooling_test(shape):
    data = keras.layers.Input(shape=shape)
    x = keras.layers.MaxPooling2D()(data)
    y = keras.layers.MaxPooling2D()(x)
    z = keras.layers.AveragePooling2D()(data)
    w = keras.layers.AveragePooling2D()(z)
    keras_model = keras.models.Model(data, z)
    verify_keras_frontend(keras_model) 

def merge_and_pool_test(shape):
    data = keras.layers.Input(shape=shape)
    x = keras.layers.MaxPooling2D()(data)
    y = keras.layers.AveragePooling2D()(data)
    out = keras.layers.Add()([x,y])
    keras_model = keras.models.Model(data, out)
    verify_keras_frontend(keras_model) 

def merge_just_conv_test():
    data = keras.layers.Input(shape=(32,3,32))
    out = keras.layers.Conv2D(8, (3, 3), padding="same")(data)
    keras_model = keras.models.Model(data, out)
    verify_keras_frontend(keras_model)

#merge_test((3,3))
#verify_keras_frontend(merge_test((10,7,4)))
#verify_keras_frontend(merge_2_test((3,3)))
#pooling_test((32,32,16))
#pooling_test((32,16,32))
#pooling_test((16,32,32))
merge_and_pool_test((32,32,32))
#merge_just_conv_test()
print("All Passed!")