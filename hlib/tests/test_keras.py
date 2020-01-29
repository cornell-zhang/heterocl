import keras
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Dense, Flatten
import numpy as np
import heterocl as hcl
import tvm
import numpy.testing as tst
import hlib
from hlib.frontend.relay_parser import relay_parser, get_relay_model

def verify_keras_frontend(keras_model, need_trans_before=True,
                          need_trans_after=True, dtype='float32', test_name=""):
    hcl.init(hcl.Float())
    assert(keras.backend.backend() == 'tensorflow')
    if(keras_model is None):
        return
    in_shapes = []
    for layer in keras_model._input_layers:
        in_shapes.append(
            tuple(
                dim if dim is not None else 1 for dim in layer.input.shape
                )
            )

    def get_keras_output(xs, dtype='float32'):
        return keras_model.predict(xs)

    def get_hcl_output(xs, dtype='float32'):
        shape_dict = {
            name: x.shape for (name, x) in zip(keras_model.input_names, xs)
            }
        return get_relay_model(keras_model, shape_dict, 'keras')

    def to_channels_first(arr):
        if len(arr.shape) > 1:
            return arr.transpose([0, -1] + list(range(1, arr.ndim - 1)))
        else:
            return arr

    def to_channels_last(arr):
        if len(arr.shape) > 1:
            return arr.transpose([0] + list(range(2, arr.ndim)) + [1])
        else:
            return arr

    xs = [
        np.random.randint(
            size=shape,
            low=1,
            high=10).astype(dtype) for shape in in_shapes]
    keras_out = get_keras_output(xs, dtype)
    inputs = [to_channels_first(x) for x in xs] if need_trans_before else xs
    f, params = get_hcl_output(inputs, dtype)
    out = []
    if isinstance(keras_out, (tuple, list)):
        for k_out in keras_out:
            out.append(hcl.asarray(np.zeros(k_out.shape)))
    else:
        out.append(hcl.asarray(np.zeros(keras_out.shape)))
    for i in range(len(inputs)):
        inputs[i] = hcl.asarray(inputs[i])

    f(*inputs, *params, *out)

    if isinstance(keras_out, (tuple, list)):
        for i in range(len(keras_out)):
            if need_trans_after:
                h_out = out[i].asnumpy()
                tst.assert_almost_equal(
                    np.reshape(
                        np.transpose(out[i].asnumpy(), (0, 1, 3, 2)),
                        keras_out[i].shape),
                    keras_out[i],
                    5)
            else:
                h_out = out[i].asnumpy()
                tst.assert_almost_equal(h_out, keras_out[i], 5)
    else:
        if(need_trans_after):
            shape = out[0].shape
            h_out = np.reshape(
                out[0].asnumpy(),
                (shape[0], shape[3], shape[1], shape[2]))
            h_out = np.transpose(h_out, [0, 2, 3, 1])
            tst.assert_almost_equal(h_out, keras_out, 5)
        else:
            shape = out[0].shape
            h_out = out[0].asnumpy()
            tst.assert_almost_equal(h_out, keras_out, 5)


def test_merge():
    def _test(shape):
        x = keras.layers.Input(shape=shape)
        y = keras.layers.Input(shape=shape)
        z = keras.layers.Input(shape=shape)
        merge_funcs = [keras.layers.Add(),
                       keras.layers.Subtract(),
                       keras.layers.Multiply(),
                       keras.layers.Maximum(),
                       keras.layers.Average(),
                       keras.layers.Concatenate(axis=1)]
        for merge_func in merge_funcs:
            if isinstance(merge_func, (keras.layers.merge.Subtract,
                                       keras.layers.merge.Dot)):
                out = merge_func([x, y])
                keras_model = keras.models.Model([x, y], out)
            else:
                out = merge_func([x, y, z])
                keras_model = keras.models.Model([x, y, z], out)
            verify_keras_frontend(keras_model, False, False)

    _test((2, 2))
    _test((10, 7, 4))


def test_merge_2():
    x = keras.layers.Input(shape=(3, 3))
    y = keras.layers.Input(shape=(3, 3))
    merge_funcs = [keras.layers.Subtract(),
                   keras.layers.Average()]
    for merge_func in merge_funcs:
        out = merge_func([x, y])
        keras_model = keras.models.Model([x, y], out)
        verify_keras_frontend(keras_model, False, False)


def test_merge_conv():
    data = keras.layers.Input(shape=(3, 3, 2))
    x = keras.layers.Conv2D(4, (3, 3), padding="same")(data)
    y = keras.layers.Conv2D(4, (3, 3), padding="same")(data)
    z = keras.layers.Conv2D(4, (3, 3), padding="same")(data)
    merge_funcs = [keras.layers.Add(),
                   keras.layers.Subtract(),
                   keras.layers.Multiply(),
                   keras.layers.Maximum(),
                   keras.layers.Average(),
                   keras.layers.Concatenate()]
    for merge_func in merge_funcs:
        if isinstance(merge_func, (keras.layers.merge.Subtract,
                                   keras.layers.merge.Dot)):
            out = merge_func([x, y])
        else:
            out = merge_func([x, y, z])
    keras_model = keras.models.Model(data, out)
    verify_keras_frontend(keras_model, True, True)


def test_pooling():
    def _test(shape):
        data = keras.layers.Input(shape=shape)
        x = keras.layers.MaxPooling2D()(data)
        y = keras.layers.MaxPooling2D()(x)
        z = keras.layers.AveragePooling2D()(y)
        w = keras.layers.AveragePooling2D()(z)
        keras_model = keras.models.Model(data, w)
        verify_keras_frontend(keras_model)

    _test((32, 32, 16))
    _test((32, 16, 32))
    _test((16, 32, 32))


def test_pooling_2():
    def _test(shape, filter, strides, padding):
        data = keras.layers.Input(shape=shape)
        x = keras.layers.MaxPooling2D(
                filter,
                strides=strides,
                padding=padding
                )(data)
        keras_model = keras.models.Model(data, x)
        verify_keras_frontend(keras_model)

    _test((4, 4, 1), (2, 2), (1, 1), "same")
    _test((4, 4, 1), (2, 2), (2, 2), "same")
    _test((5, 5, 1), (4, 4), (1, 1), "same")
    _test((5, 5, 1), (4, 4), (2, 2), "same")
    _test((4, 4, 1), (2, 2), (1, 1), "valid")
    _test((4, 4, 1), (2, 2), (2, 2), "valid")
    _test((5, 5, 1), (4, 4), (1, 1), "valid")
    _test((5, 5, 1), (4, 4), (2, 2), "valid")


def test_batch_norm():
    def _test(shape, axis):
        data = keras.layers.Input(shape=shape)
        x = keras.layers.BatchNormalization(axis=axis)(data)
        y = keras.layers.BatchNormalization(axis=axis)(x)
        keras_model = keras.models.Model(data, y)
        verify_keras_frontend(keras_model, False, False)

    _test((4, 4), 1)
    _test((4, 4), 2)
    _test((4, 4), -1)
    _test((4, 4, 4), -1)


def test_merge_and_pool():
    def _test(shape):
        data = keras.layers.Input(shape=shape)
        x = keras.layers.MaxPooling2D()(data)
        w = keras.layers.MaxPooling2D()(x)
        y = keras.layers.AveragePooling2D()(data)
        z = keras.layers.AveragePooling2D()(y)
        out = keras.layers.Add()([w, z])
        keras_model = keras.models.Model(data, out)
        verify_keras_frontend(keras_model, True, True)

    _test((16, 8, 4))
    _test((8, 8, 8))

def test_merge_out_tup():
    data = keras.layers.Input(shape=(4, 4, 4))
    x = keras.layers.MaxPooling2D()(data)
    z = keras.layers.MaxPooling2D()(x)
    y = keras.layers.AveragePooling2D()(data)
    w = keras.layers.AveragePooling2D()(y)
    keras_model = keras.models.Model(data, [z, w])
    verify_keras_frontend(keras_model)


def test_merge_just_conv():
    data = keras.layers.Input(shape=(4, 4, 3))
    out = keras.layers.Conv2D(
            3, (2, 2), padding="same", use_bias=False
            )(data)
    keras_model = keras.models.Model(data, out)
    verify_keras_frontend(keras_model, True, True)


"""
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
        verify_keras_frontend(keras_model,False,False)
"""

def test_sequential():
    keras_model = keras.models.Sequential([
        keras.layers.Dense(16, input_dim=32, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    verify_keras_frontend(keras_model, False, False)


def test_simple_pool():
    data = keras.layers.Input(shape=(9, 9, 3))
    # maxpool
    x = keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(
        keras_model,
        need_trans_before=True,
        need_trans_after=True)
    # avgpool
    y = keras.layers.AveragePooling2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='valid')(data)
    keras_model = keras.models.Model(data, y)
    verify_keras_frontend(
        keras_model,
        need_trans_before=True,
        need_trans_after=True)


def test_reshape():
    # input_shape len is 3, target_shape len is 3
    data = keras.layers.Input(shape=(32, 32, 3))
    x = keras.layers.Reshape(target_shape=(16, 64, 3))(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model, False, False)
    # input_shape len is 3, target_shape len is 2
    data = keras.layers.Input(shape=(32, 8, 3))
    x = keras.layers.Reshape(target_shape=(256, 3))(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model, False, False)
    # input_shape len is 2, target_shape len is 3
    data = keras.layers.Input(shape=(256, 3))
    x = keras.layers.Reshape(target_shape=(8, 32, 3))(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model, False, False)
    # input_shape len is 2, target_shape len is 1
    data = keras.layers.Input(shape=(2, 8))
    x = keras.layers.Reshape(target_shape=(16,))(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model, False, False)
    # input_shape len is 1, target_shape len is 2
    data = keras.layers.Input(shape=(16,))
    x = keras.layers.Reshape(target_shape=(4, 4))(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model, False, False)
    # input_shape len is 2, target_shape len is 2
    data = keras.layers.Input(shape=(2, 8))
    x = keras.layers.Reshape(target_shape=(4, 4))(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model, False, False)


def test_rnn():
    data = keras.layers.Input(shape=(1, 32))
    names = ["lstm", "rnn", "gru"]
    i = 0
    rnn_funcs = [keras.layers.LSTM(units=16, return_state=False,
                                   recurrent_activation='sigmoid', activation='tanh'),
                 keras.layers.SimpleRNN(units=16, return_state=False,
                                        activation='tanh'),
                 keras.layers.GRU(units=16, return_state=False,
                                  recurrent_activation='sigmoid', activation='tanh')]
    for rnn_func in rnn_funcs:
        x = rnn_func(data)
        keras_model = keras.models.Model(data, x)
        verify_keras_frontend(keras_model, False, False, test_name=names[i])
        i += 1


def test_dense():
    data = keras.layers.Input(shape=(32, 32, 1))
    x = keras.layers.Flatten()(data)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(
        10,
        activation='relu',
        kernel_initializer='uniform'
        )(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model, True, False)


def test_dense_2():
    data = keras.layers.Input(shape=(32, 32, 1))
    x = keras.layers.Flatten()(data)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(
        10,
        activation='softmax',
        kernel_initializer='uniform'
        )(x)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model, True, False)


def test_conv_code():
    input_1 = hcl.placeholder(shape=(1, 3, 3, 3))
    param_1 = hcl.placeholder(shape=(3, 3, 3, 3))
    param_2 = hcl.placeholder(shape=(3,))
    padding = []
    strides = []
    dilation = []
    axis = 1
    for i in range(2):
        padding.append(tvm.expr.IntImm(dtype='int64', value=1))
        strides.append(tvm.expr.IntImm(dtype='int32', value=1))
        dilation.append(tvm.expr.IntImm(dtype='int32', value=1))

    def func(_in, filt, bias):
        i_0 = hlib.op.nn.conv2d(_in, filt, padding=padding,
                                strides=strides, dilation=dilation)
        return hlib.op.nn.bias_add(i_0, bias, axis=axis)
    s = hcl.create_schedule([input_1, param_1, param_2], func)
    f = hcl.build(s)
    _in = hcl.asarray(np.random.randint(10, size=(1, 3, 3, 3)))
    filt = hcl.asarray(np.random.randint(10, size=(3, 3, 3, 3)))
    bias = hcl.asarray(np.random.randint(10, size=(3,)))
    out = hcl.asarray(np.zeros((1, 3, 3, 3)))
    f(_in, filt, bias, out)


def test_forward_multi_inputs():
    data1 = keras.layers.Input(shape=(32, 32, 3))
    data2 = keras.layers.Input(shape=(32, 32, 3))
    x = keras.layers.Conv2D(8, (3, 3), padding="same")(data1)
    y = keras.layers.Conv2D(8, (3, 3), padding="same")(data2)
    z = keras.layers.Average()([x, y])
    z = keras.layers.GlobalAveragePooling2D()(z)
    keras_model = keras.models.Model([data1, data2], z)
    verify_keras_frontend(keras_model, True, False)


def test_forward_multi_outputs():
    data = keras.layers.Input(shape=(32, 32, 3))
    x = keras.layers.Conv2D(8, (3, 3), padding="same")(data)
    x = keras.layers.GlobalAveragePooling2D()(x)
    y = keras.layers.Conv2D(8, (3, 3), padding="same")(data)
    y = keras.layers.GlobalAveragePooling2D()(y)
    z = keras.layers.Conv2D(8, (3, 3), padding="same")(data)
    z = keras.layers.GlobalMaxPooling2D()(z)
    w = keras.layers.Conv2D(8, (3, 3), padding="same")(data)
    w = keras.layers.GlobalMaxPooling2D()(w)
    keras_model = keras.models.Model(data, [x, y, z, w])
    verify_keras_frontend(keras_model, True, False)


def test_reuse_layers():
    # reuse conv2d
    data = keras.layers.Input(shape=(32, 32, 3))
    conv2d = keras.layers.Conv2D(8, (3, 3), padding="same")
    x = conv2d(data)
    y = conv2d(data)
    z = keras.layers.Add()([x, y])
    z = keras.layers.GlobalAveragePooling2D()(z)
    keras_model = keras.models.Model(data, z)
    verify_keras_frontend(keras_model, True, False)
    # reuse add
    data = keras.layers.Input(shape=(32, 32, 3))
    x = keras.layers.Conv2D(8, (3, 3), padding="same")(data)
    add = keras.layers.Add()
    x = add([x, x])
    x = add([x, x])
    z = keras.layers.GlobalAveragePooling2D()(x)
    keras_model = keras.models.Model(data, z)
    verify_keras_frontend(keras_model, True, False)


def test_for_paper():
    in1 = keras.layers.Input((4, 3, 3))
    act0 = keras.layers.Activation('sigmoid')(in1)
    act1 = keras.layers.Activation('relu')(in1)
    act2 = keras.layers.Activation('tanh')(in1)
    add1 = keras.layers.Add()([act0, act1, act2])
    keras_model = keras.models.Model(in1, add1)
    verify_keras_frontend(keras_model, False, False)


def test_multiple_reuse():
    in1 = keras.layers.Input((4, 3, 3))
    act0 = keras.layers.Activation('sigmoid')(in1)
    act1 = keras.layers.ReLU()(act0)
    add1 = keras.layers.Add()([act0, act1])
    act2 = keras.layers.ReLU()(add1)
    add2 = keras.layers.Add()([act1, act2])
    add3 = keras.layers.Add()([act1, add2])
    keras_model = keras.models.Model(in1, add3)
    verify_keras_frontend(keras_model, False, False)


def test_forward_conv():
    data = keras.layers.Input(shape=(4, 4, 2))
    conv_funcs = [keras.layers.Conv2D(filters=10, kernel_size=(3, 3), strides=(2, 2), padding='same'),
                  keras.layers.Conv2D(filters=10, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False),
                  keras.layers.Conv2D(filters=10, kernel_size=(3, 3), dilation_rate=(2, 2), padding='same'),
                  keras.layers.Conv2D(filters=10, kernel_size=(1, 1), strides=(2, 2), padding='same'),
                  keras.layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same'),
                  keras.layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same'),
                  #keras.layers.Conv2DTranspose(filters=10, kernel_size=(3, 3), padding='valid'), can be implemented later
                  keras.layers.SeparableConv2D(filters=10, kernel_size=(3, 3), padding='same')]
    for conv_func in conv_funcs:
        x = conv_func(data)
        keras_model = keras.models.Model(data, x)
        verify_keras_frontend(keras_model, True, True)


def test_depthwise_conv():
    data = keras.layers.Input(shape=(4, 4, 3))
    x = keras.layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model, True, True)


def test_separable_conv():
    data = keras.layers.Input(shape=(4, 4, 3))
    x = keras.layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(data)
    keras_model = keras.models.Model(data, x)
    verify_keras_frontend(keras_model, True, True)


def test_forward_activations():
    data = keras.layers.Input(shape=(100,))
    act_funcs = [keras.layers.Activation('softmax'),
                 keras.layers.Softmax(),
                 keras.layers.Softmax(axis=-1),
                 keras.layers.Softmax(axis=1),
                 #keras.layers.Softmax(axis=2), Relay is incorrect
                 #keras.layers.Softmax(axis=3),
                 keras.layers.Activation('softplus'),
                 keras.layers.Activation('relu'),
                 #keras.layers.Activation('softsign'), FIX THIS!
                 keras.layers.Activation('hard_sigmoid'),
                 keras.layers.Activation('sigmoid'),
                 keras.layers.Activation('tanh'),
                 keras.layers.Activation('linear'),
                 keras.layers.Activation('selu'),
                 keras.layers.ReLU(),
                 keras.layers.ReLU(max_value=6.),
                 #keras.layers.ReLU(max_value=6., threshold=0.), Relay IR formats these terribly
                 #keras.layers.ReLU(max_value=6., threshold=1.),
                 #keras.layers.ReLU(max_value=6., threshold=1., negative_slope=0.),
                 #keras.layers.ReLU(max_value=6., threshold=1., negative_slope=0.5),
                 #keras.layers.ReLU(max_value=6., threshold=1., negative_slope=1.),
                 keras.layers.LeakyReLU(alpha=0.3),
                 #keras.layers.PReLU(weights=np.random.rand(1, 8, 3, 3)), Relay IR formats this terribly
                 keras.layers.ELU(alpha=0.5),
                 keras.layers.ThresholdedReLU(theta=0.5)]
    for act_func in act_funcs:
        x = act_func(data)
        keras_model = keras.models.Model(data, x)
        verify_keras_frontend(keras_model, False, False)


def test_cifar10():
    num_classes = 10
    model = keras.models.Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(16, 16, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    input_layer = keras.layers.Input(batch_shape=model.layers[0].input_shape)
    prev_layer = input_layer
    for layer in model.layers:
        prev_layer = layer(prev_layer)

    model = keras.models.Model([input_layer], [prev_layer])
    verify_keras_frontend(model, True, False)


def test_forward_vgg16():
    keras_model = keras.applications.VGG16(include_top=True, weights='imagenet',
                                           input_shape=(224, 224, 3), classes=1000)
    verify_keras_frontend(keras_model, True, False)


def test_forward_xception():
    keras_model = keras.applications.Xception(include_top=True, weights='imagenet',
                                              input_shape=(299, 299, 3), classes=1000)
    verify_keras_frontend(keras_model, True, False)


def test_forward_resnet50():
    keras_model = keras.applications.ResNet50(include_top=True, weights='imagenet',
                                              input_shape=(224, 224, 3), classes=1000)
    verify_keras_frontend(keras_model, True, False)


def test_forward_mobilenet():
    keras_model = keras.applications.MobileNet(include_top=True, weights='imagenet',
                                               input_shape=(224, 224, 3), classes=1000)
    verify_keras_frontend(keras_model, True, False, 'float64')
