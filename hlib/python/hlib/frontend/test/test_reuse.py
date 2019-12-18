import keras
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Dense, Flatten
import numpy as np
import heterocl as hcl
import tvm
from tvm import relay
import tvm.relay.frontend as relay_front
import numpy.testing as tst
from frontend.relay_parser import relay_parser, get_relay_model, model_extent
import hlib

def test_multiple_reuse():
    in1 = keras.layers.Input((4,3,3))
    act1 = keras.layers.ReLU()(in1)
    add1 = keras.layers.Add()([in1,act1])
    act2 = keras.layers.ReLU()(add1)
    add2 = keras.layers.Add()([act1,act2])
    add3 = keras.layers.Add()([act1,add2])
    keras_model = keras.models.Model(in1,add3)
    in_shapes=[]
    for layer in keras_model._input_layers:
        in_shapes.append(tuple(dim.value if dim.value is not None else 1 for dim in layer.input.shape))
    shape_dict = {name: x for (name, x) in zip(keras_model.input_names, in_shapes)}
    module, params = relay_front.from_keras(keras_model,shape_dict)
    return module

module = test_multiple_reuse()
node_map = {}
body = module.functions[module.global_var_map_['main']]
length = model_extent(body.body,True,node_map)