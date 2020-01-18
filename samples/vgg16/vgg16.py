import heterocl as hcl
import heterocl.tvm as tvm
import hlib
import os
import numpy as np
from collections import OrderedDict

evaluation = True
dtype = "float32"
hcl.config.init_dtype = dtype
batch_size = 10

# plarform information
# target = "llvm"
# tool = hcl.tool.sdsoc("syn")
tool = hcl.tool.sdaccel("syn")
target = hcl.platform.aws_f1(tool)

names = [ 
  'conv1_1_weight', 'conv1_1_bias', 
  'conv1_2_weight', 'conv1_2_bias', 
  'conv2_1_weight', 'conv2_1_bias', 
  'conv2_2_weight', 'conv2_2_bias', 
  'conv3_1_weight', 'conv3_1_bias', 
  'conv3_2_weight', 'conv3_2_bias', 
  'conv3_3_weight', 'conv3_3_bias', 
  'conv4_1_weight', 'conv4_1_bias', 
  'conv4_2_weight', 'conv4_2_bias', 
  'conv4_3_weight', 'conv4_3_bias', 
  'conv5_1_weight', 'conv5_1_bias', 
  'conv5_2_weight', 'conv5_2_bias',
  'conv5_3_weight', 'conv5_3_bias', 
  'fc6_weight', 'fc6_bias', 
  'fc7_weight', 'fc7_bias', 
  'fc8_weight', 'fc8_bias', 
]

def build_vgg(*args):

    # create reusable modules (input ptrs)
    in_shape = out_shape = (10,128,224,224)
    w_shape = (512,512,3,3,) # (out, in, x, y)
    @hcl.def_([in_shape,out_shape,w_shape,(1,),(1,),(1,),(1,),(1,),(1,)])
    def conv_layer(img_in, img_out, weight, batch, in_num, out_num, width, k, pad):
        # pad the input images symetrically 
        out = hcl.scalar(width[0] + 2 * pad[0], "pad_width")
        Input = hcl.compute((batch[0], in_num[0], out[0], out[0]), 
            lambda nn, ff, yy, xx: hcl.select(
                hcl.and_(yy >= pad[0], xx >=pad[0], yy < width[0] + pad[0], xx < width[0] + pad[0]), 
                img_in[nn, ff, yy-pad[0], xx-pad[0]], 0), name="padded")
        rc = hcl.reduce_axis(0, in_num[0], name="in_num")
        ry = hcl.reduce_axis(0, k[0], name="ry")
        rx = hcl.reduce_axis(0, k[0], name="rx")
        sum = hcl.reducer(0, lambda x, y: x + y, dtype)
        # update over (batch, out_num, out, out)
        hcl.update(img_out, 
            lambda nn, ff, yy, xx: sum(
                Input[nn, rc, yy + ry, xx + rx] *
                weight[ff, rc, ry, rx], axis=[rc, ry, rx]),
            name = "conv_layer")
        
# download parameters
import mxnet as mx
from mxnet import nd, gluon
path='http://data.mxnet.io/models/imagenet/'
[mx.test_utils.download(path+'vgg/vgg16-symbol.json'),
 mx.test_utils.download(path+'vgg/vgg16-0000.params'),
 mx.test_utils.download(path+'synset.txt')]
sym, arg_params, aux_params = mx.model.load_checkpoint('vgg16', 0)

holders, values = [], []
params = arg_params.copy()
params.update(aux_params)
for name in names:
    val = params[name].asnumpy()
    ph  = hcl.placeholder(val.shape, name)
    holders.append(ph)
    values.append(hcl.asarray(val, dtype=hcl.Float()))

# build the function
input_image = hcl.placeholder((batch_size, 128, 224, 224), "input_image")
pred = hcl.placeholder((batch_size, 1000), "pred")
arg_list = [input_image, pred] + holders
scheme = hcl.create_scheme(arg_list, build_vgg)
s = hcl.create_schedule_from_scheme(scheme)
print(hcl.lower(s))
