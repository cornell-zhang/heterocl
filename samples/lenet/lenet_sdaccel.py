import heterocl as hcl
import numpy as np
from lenet_main import *

batch_size = 50

# f = build_lenet_inf(batch_size, 'vhls_csim')
f = build_lenet_inf(batch_size, 'sdaccel_sw_emu')

mnist = mx.test_utils.get_mnist()
correct_sum = 0

for i in range(50 // batch_size):
    label = mnist['test_label'][i*batch_size:(i+1)*batch_size]
    input_image_np = mnist['test_data'][i*batch_size:(i+1)*batch_size]
    input_image_hcl = hcl.asarray(input_image_np)
    output_hcl = hcl.asarray(np.zeros((batch_size,10)))
    f(input_image_hcl, weight_conv1_hcl, weight_conv2_hcl, weight_fc1_hcl, weight_fc2_hcl, output_hcl)
    prediction = np.argmax(output_hcl.asnumpy(), axis=1)
    correct_sum += np.sum(np.equal(prediction, label))

print(str(qtype1) + ", " + str(qtype2) + ": Accuracy over 10000 test images is: {}".format(correct_sum / 10000.))
assert correct_sum == 9882
