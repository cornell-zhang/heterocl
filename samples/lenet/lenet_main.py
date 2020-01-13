"""
HeteroCL Tutorial : LeNet Inference
===================================

**Author**: Yi-Hsiang Lai (seanlatias@github)

Build the LeNet inference model by using hlib. In this tutorial, we demonstrate
how we can apply inference-time quantization to an existing model.
"""
import heterocl as hcl
import hlib
import numpy as np

###############################################################################
# Initially we let every operation to be a floating-point operation
hcl.init(hcl.Float())

###############################################################################
# You can define your own layer without using the one provided in hlib
def softmax(out, x):
    assert len(x.shape) == 2, "only support 2-dim softmax"
    m, n = x.shape
    k = hcl.reduce_axis(0, n)
    max_elem = hcl.compute((m,), lambda i: hcl.max(x[i, k], axis=k))
    k = hcl.reduce_axis(0, n)
    expsum = hcl.compute((m,),
            lambda i: hcl.sum(hcl.exp(x[i, k] - max_elem[i]), axis=k))
    return hcl.update(out,
            lambda i, j: hcl.exp(x[i, j] - max_elem[i]) / expsum[i])

###############################################################################
# The main function for build the LeNet inference model
def build_lenet(input_image, weight_conv1, weight_conv2,
                weight_fc1, weight_fc2, lenet):
    # first conv
    conv1 = hlib.op.nn.conv2d_nchw(input_image, weight_conv1)
    tanh1 = hlib.op.math.tanh(conv1, "tanh1")
    pool1 = hlib.op.nn.max_pool(tanh1, kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = hlib.op.nn.conv2d_nchw(pool1, weight_conv2)
    tanh2 = hlib.op.math.tanh(conv2, "tanh2")
    pool2 = hlib.op.nn.max_pool(tanh2, kernel=(2,2), stride=(2,2))
    # first fc
    flat = hlib.op.nn.flatten(pool2)
    fc1 = hlib.op.nn.dense(flat, weight_fc1)
    tanh3 = hlib.op.math.tanh(fc1, "tanh3")
    # second fc
    fc2 =  hlib.op.nn.dense(tanh3, weight_fc2)
    # loss
    return softmax(lenet, fc2)

###############################################################################
# Download the dataset from mxnet
import mxnet as mx
# download pretrained lenet model
mx.gluon.utils.download('https://gist.githubusercontent.com/Huyuwei/dc00ce83f537914c64a204133d23b019/raw/79af41e7c8ba9120ea7f35fb1d0484b65bccd54f/lenet-0010.params')
mx.gluon.utils.download('https://gist.githubusercontent.com/Huyuwei/dc00ce83f537914c64a204133d23b019/raw/79af41e7c8ba9120ea7f35fb1d0484b65bccd54f/lenet-symbol.json')
sym, arg_params, aux_params = mx.model.load_checkpoint('lenet', 10)
# get weights
weight_conv1_np = arg_params['convolution0_weight'].asnumpy()
weight_conv2_np = arg_params['convolution1_weight'].asnumpy()
weight_fc1_np = arg_params['fullyconnected0_weight'].asnumpy()
weight_fc2_np = arg_params['fullyconnected1_weight'].asnumpy()

###############################################################################
# Define the quantized data type and run the inference
qtype1 = hcl.Fixed(16, 14)
qtype2 = hcl.Fixed(16, 14)
correct_sum = 0
batch_size = 1000
mnist = mx.test_utils.get_mnist()

###############################################################################
# In this example, we quantize the weights to `qtype1` and the activations to
# `qtype2`. To quantize the placeholders, simply specify the `dtype` field. For
# the internal tensors, we use `hcl.quantize` API.
def build_lenet_inf(batch_size=batch_size, target=None):
    # set up input/output placeholders
    input_image = hcl.placeholder((batch_size, 1, 28, 28), "input_image")
    weight_conv1 = hcl.placeholder((20, 1, 5, 5), "weight_conv1", qtype1)
    weight_conv2 = hcl.placeholder((50, 20, 5, 5), "weight_conv2", qtype1)
    weight_fc1 = hcl.placeholder((500, 800), "weight_fc1", qtype1)
    weight_fc2 = hcl.placeholder((10, 500), "weight_fc2", qtype1)
    lenet = hcl.placeholder((batch_size, 10), "lenet")
    # create a quantization scheme
    scheme = hcl.create_scheme(
            [input_image, weight_conv1, weight_conv2,
             weight_fc1, weight_fc2, lenet], build_lenet)
    # quantize the three activation layers
    scheme.quantize(
            [build_lenet.tanh1, build_lenet.tanh2, build_lenet.tanh3], qtype2)
    s = hcl.create_schedule_from_scheme(scheme)
    return hcl.build(s, target=target)

f = build_lenet_inf()

###############################################################################
# Prepare the numpy arrays for testing. Remember that we need to set the input
# tensors with the same type as the placeholders
weight_conv1_hcl = hcl.asarray(weight_conv1_np, dtype=qtype1)
weight_conv2_hcl = hcl.asarray(weight_conv2_np, dtype=qtype1)
weight_fc1_hcl = hcl.asarray(weight_fc1_np, dtype=qtype1)
weight_fc2_hcl = hcl.asarray(weight_fc2_np, dtype=qtype1)

for i in range(10000 // batch_size):
    label = mnist['test_label'][i*batch_size:(i+1)*batch_size]
    input_image_np = mnist['test_data'][i*batch_size:(i+1)*batch_size]
    input_image_hcl = hcl.asarray(input_image_np)
    output_hcl = hcl.asarray(np.zeros((batch_size,10)))
    f(input_image_hcl, weight_conv1_hcl, weight_conv2_hcl,
            weight_fc1_hcl, weight_fc2_hcl, output_hcl)
    prediction = np.argmax(output_hcl.asnumpy(), axis=1)
    correct_sum += np.sum(np.equal(prediction, label))

###############################################################################
# Print the result

print("Testing accuracy: {}".format(correct_sum / 10000.))

# remove downloaded files
import os
os.remove("t10k-images-idx3-ubyte.gz")
os.remove("t10k-labels-idx1-ubyte.gz")
os.remove("train-images-idx3-ubyte.gz")
os.remove("train-labels-idx1-ubyte.gz")
os.remove("lenet-0010.params")
os.remove("lenet-symbol.json")

assert correct_sum == 9882
