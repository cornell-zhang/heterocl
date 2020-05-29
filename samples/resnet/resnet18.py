import heterocl as hcl
import heterocl.tvm as tvm
import hlib
import os
import numpy as np

evaluation = True
dtype = "float32"
hcl.config.init_dtype = dtype
batch_size = 1

# plarform information
# target = "llvm"
tool = hcl.tool.sdsoc("syn")
# tool = hcl.tool.sdaccel("syn")
target = hcl.platform.aws_f1(tool)

unit1_names = [
    's1_u1_b1', 's1_u1_r1', 's1_u1_c1',
    's1_u1_b2', 's1_u1_r2', 's1_u1_c2', 's1_u1_sc', 's1_u1_add', 
    's2_u1_b1', 's2_u1_r1', 's2_u1_c1',
    's2_u1_b2', 's2_u1_r2', 's2_u1_c2', 's2_u1_sc', 's2_u1_add', 
    's3_u1_b1', 's3_u1_r1', 's3_u1_c1',
    's3_u1_b2', 's3_u1_r2', 's3_u1_c2', 's3_u1_sc', 's3_u1_add', 
    's4_u1_b1', 's4_u1_r1', 's4_u1_c1',
    's4_u1_b2', 's4_u1_r2', 's4_u1_c2', 's4_u1_sc', 's4_u1_add'
]

unit2_names = [
    's1_u2_b1', 's1_u2_r1', 's1_u2_c1',
    's1_u2_b2', 's1_u2_r2', 's1_u2_c2', 's1_u2_add', 
    's2_u2_b1', 's2_u2_r1', 's2_u2_c1',
    's2_u2_b2', 's2_u2_r2', 's2_u2_c2', 's2_u2_add', 
    's3_u2_b1', 's3_u2_r1', 's3_u2_c1',
    's3_u2_b2', 's3_u2_r2', 's3_u2_c2', 's3_u2_add',
    's4_u2_b1', 's4_u2_r1', 's4_u2_c1',
    's4_u2_b2', 's4_u2_r2', 's4_u2_c2', 's4_u2_add' 
]

fnames = [
    'bn', 'conv0', 'bn0', 'relu0', 'pool0' 
]

enames = [
    'bn1', 'relu1', 'pool1', 'fc1'
]

name_pool = unit1_names + unit2_names + enames + fnames

def build_resnet(*args):

    def unit1(stage, *args):
        # with projection layer 
        # input -> bn -> relu -> (conv+bn+relu+conv / conv) -> add
        assert len(args) == 12, "# of parames less than 12"
        input_fm, bn1_beta, bn1_gamma, bn1_mean, bn1_var, \
            conv1_weight, bn2_beta, bn2_gamma, bn2_mean, bn2_var, \
            conv2_weight, convsc_weight = args
        if stage == 1:
           strides = [[1,1], [1,1], [1,1]]
        else:
           strides = [[2,2], [1,1], [2,2]]

        base = (stage-1) * 8
        bn1 = hlib.nn.batch_norm(input_fm, 
                              bn1_beta, 
                              bn1_gamma, 
                              bn1_mean,
                              bn1_var,
                              unit1_names[base+0])
        relu1 = hlib.nn.relu(bn1, unit1_names[base+1]) 

        # residual path
        conv1 = hlib.nn.conv2d_nchw(relu1, 
                                  conv1_weight, 
                                  name=unit1_names[base+2],
                                  stride=strides[0],
                                  padding=[[1,1],[1,1]])
        bn2 = hlib.nn.batch_norm(conv1, 
                              bn2_beta, 
                              bn2_gamma,
                              bn2_mean,
                              bn2_var,
                              unit1_names[base+3])
        relu2 = hlib.nn.relu(bn2, unit1_names[base+4]) 
        conv2 = hlib.nn.conv2d_nchw(relu2, 
                                  conv2_weight, 
                                  name=unit1_names[base+5],
                                  stride=strides[1],
                                  padding=[[1,1],[1,1]])
        # projection path
        convsc = hlib.nn.conv2d_nchw(relu1, 
                                   convsc_weight, 
                                   name=unit1_names[base+6],
                                   stride=strides[2])
        # element-wise add
        plus = hlib.nn.tensoradd(convsc, 
                               conv2,
                               unit1_names[base+7])
        return plus

    def unit2(stage, *args):
        # without projection layer 
        # (bn+relu+conv+bn+relu+conv / reference) -> add
        assert len(args) == 11, "# of parames less than 11"
        input_fm, bn1_beta, bn1_gamma, bn1_mean, bn1_var, \
            conv1_weight, bn2_beta, bn2_gamma, \
            bn2_mean, bn2_var, conv2_weight = args

        base = (stage - 1) * 7

        # residual path
        padding = [[1,1], [1,1]]
        bn1 = hlib.nn.batch_norm(input_fm, 
                              bn1_beta, 
                              bn1_gamma,
                              bn1_mean,
                              bn1_var,
                              unit2_names[base+0])
        relu1 = hlib.nn.relu(bn1, unit2_names[base+1]) 
        conv1 = hlib.nn.conv2d_nchw(relu1, 
                                  conv1_weight, 
                                  name=unit2_names[base+2],
                                  padding=padding)

        bn2 = hlib.nn.batch_norm(conv1, 
                              bn2_beta, 
                              bn2_gamma,
                              bn2_mean,
                              bn2_var,
                              unit2_names[base+3])
        relu2 = hlib.nn.relu(bn2, unit2_names[base+4]) 
        conv2 = hlib.nn.conv2d_nchw(relu2, 
                                  conv2_weight, 
                                  name=unit2_names[base+5],
                                  padding=padding)

        # element-wise add
        plus = hlib.nn.tensoradd(input_fm, 
                               conv2,
                               unit2_names[base+6])
        return plus
        
    # before 1st stage
    input_image, resnet, phs = args[0], args[1], args[2:11] 
    bn_beta, bn_gamma, bn_mean, bn_var, \
        conv0_weight, bn0_beta, bn0_gamma, bn0_mean, bn0_var = phs
    bn_data = hlib.nn.batch_norm(input_image, 
                              bn_beta, 
                              bn_gamma, 
                              bn_mean,
                              bn_var,
                              fnames[0])
    conv0 = hlib.nn.conv2d_nchw(bn_data, 
                              conv0_weight, 
                              name=fnames[1],
                              stride=[2, 2],
                              padding=[[3, 3], [3, 3]])
    bn0   = hlib.nn.batch_norm(conv0, 
                            bn0_beta, 
                            bn0_gamma, 
                            bn0_mean, 
                            bn0_var,
                            fnames[2])
    relu0 = hlib.nn.relu(bn0, fnames[3])
    pool0 = hlib.nn.max_pool(relu0, 
                           kernel=(3,3), 
                           stride=(2,2),
                           padding=[[1,1],[1,1]],
                           name=fnames[4])

    # first stage unit 1/2
    plus0 = unit1(1, pool0, *args[11:22])
    plus1 = unit2(1, plus0, *args[22:32])

    # second stage unit 1/2
    plus2 = unit1(2, plus1, *args[32:43])
    plus3 = unit2(2, plus2, *args[43:53])

    # third stage unit 1/2
    plus4 = unit1(3, plus3, *args[53:64])
    plus5 = unit2(3, plus4, *args[64:74])

    # fourth stage unit 1/2
    plus6 = unit1(4, plus5, *args[74:85])
    plus7 = unit2(4, plus6, *args[85:95])

    # after fourth stage
    bn1_beta, bn1_gamma, bn1_mean, bn1_var, \
        fc_weight, fc_bias = args[-6:]
    bn1   = hlib.nn.batch_norm(plus7, 
                            bn1_beta, 
                            bn1_gamma,
                            bn1_mean,
                            bn1_var,
                            enames[0])
    relu1 = hlib.nn.relu(bn1, enames[1])
    pool1 = hlib.nn.max_pool(relu1, 
                           kernel=(7,7), 
                           stride=(1,1),
                           name=enames[2])
    flatten0 = hlib.nn.flatten(pool1)
    fc1 = hlib.nn.dense(flatten0, 
                      fc_weight, 
                      bias=fc_bias,
                      name=enames[3])
    # loss function
    return hlib.nn.softmax(resnet, fc1)

# -------------------------------
# download restnet parameters
# -------------------------------
import mxnet as mx
from mxnet import nd, gluon
path='http://data.mxnet.io/models/imagenet/'
[mx.test_utils.download(path+'resnet/18-layers/resnet-18-0000.params'),
 mx.test_utils.download(path+'resnet/18-layers/resnet-18-symbol.json'),
 mx.test_utils.download(path+'synset.txt')]
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-18', 0)

# -------------------------------
# get weights and bn parameters
# -------------------------------
before = [
  'bn_data_beta',
  'bn_data_gamma',
  'bn_data_moving_mean',
  'bn_data_moving_var',
  'conv0_weight',
  'bn0_beta',
  'bn0_gamma',
  'bn0_moving_mean',
  'bn0_moving_var'
]

stage1_unit1 = [
  'stage1_unit1_bn1_beta',
  'stage1_unit1_bn1_gamma',
  'stage1_unit1_bn1_moving_mean',
  'stage1_unit1_bn1_moving_var',
  'stage1_unit1_conv1_weight',
  'stage1_unit1_bn2_beta',
  'stage1_unit1_bn2_gamma',
  'stage1_unit1_bn2_moving_mean',
  'stage1_unit1_bn2_moving_var',
  'stage1_unit1_conv2_weight',
  'stage1_unit1_sc_weight'
]

stage1_unit2 = [
  'stage1_unit2_bn1_beta',
  'stage1_unit2_bn1_gamma',
  'stage1_unit2_bn1_moving_mean',
  'stage1_unit2_bn1_moving_var',
  'stage1_unit2_conv1_weight',
  'stage1_unit2_bn2_beta',
  'stage1_unit2_bn2_gamma',
  'stage1_unit2_bn2_moving_mean',
  'stage1_unit2_bn2_moving_var',
  'stage1_unit2_conv2_weight'
]

stage2_unit1 = [
  'stage2_unit1_bn1_beta',
  'stage2_unit1_bn1_gamma',
  'stage2_unit1_bn1_moving_mean',
  'stage2_unit1_bn1_moving_var',
  'stage2_unit1_conv1_weight',
  'stage2_unit1_bn2_beta',
  'stage2_unit1_bn2_gamma',
  'stage2_unit1_bn2_moving_mean',
  'stage2_unit1_bn2_moving_var',
  'stage2_unit1_conv2_weight',
  'stage2_unit1_sc_weight'
]

stage2_unit2 = [
  'stage2_unit2_bn1_beta',
  'stage2_unit2_bn1_gamma',
  'stage2_unit2_bn1_moving_mean',
  'stage2_unit2_bn1_moving_var',
  'stage2_unit2_conv1_weight',
  'stage2_unit2_bn2_beta',
  'stage2_unit2_bn2_gamma',
  'stage2_unit2_bn2_moving_mean',
  'stage2_unit2_bn2_moving_var',
  'stage2_unit2_conv2_weight'
]

stage3_unit1 = [
  'stage3_unit1_bn1_beta',
  'stage3_unit1_bn1_gamma',
  'stage3_unit1_bn1_moving_mean',
  'stage3_unit1_bn1_moving_var',
  'stage3_unit1_conv1_weight',
  'stage3_unit1_bn2_beta',
  'stage3_unit1_bn2_gamma',
  'stage3_unit1_bn2_moving_mean',
  'stage3_unit1_bn2_moving_var',
  'stage3_unit1_conv2_weight',
  'stage3_unit1_sc_weight'
]

stage3_unit2 = [
  'stage3_unit2_bn1_beta',
  'stage3_unit2_bn1_gamma',
  'stage3_unit2_bn1_moving_mean',
  'stage3_unit2_bn1_moving_var',
  'stage3_unit2_conv1_weight',
  'stage3_unit2_bn2_beta',
  'stage3_unit2_bn2_gamma',
  'stage3_unit2_bn2_moving_mean',
  'stage3_unit2_bn2_moving_var',
  'stage3_unit2_conv2_weight'
]

stage4_unit1 = [
  'stage4_unit1_bn1_beta',
  'stage4_unit1_bn1_gamma',
  'stage4_unit1_bn1_moving_mean',
  'stage4_unit1_bn1_moving_var',
  'stage4_unit1_conv1_weight',
  'stage4_unit1_bn2_beta',
  'stage4_unit1_bn2_gamma',
  'stage4_unit1_bn2_moving_mean',
  'stage4_unit1_bn2_moving_var',
  'stage4_unit1_conv2_weight',
  'stage4_unit1_sc_weight'
]

stage4_unit2 = [
  'stage4_unit2_bn1_beta',
  'stage4_unit2_bn1_gamma',
  'stage4_unit2_bn1_moving_mean',
  'stage4_unit2_bn1_moving_var',
  'stage4_unit2_conv1_weight',
  'stage4_unit2_bn2_beta',
  'stage4_unit2_bn2_gamma',
  'stage4_unit2_bn2_moving_mean',
  'stage4_unit2_bn2_moving_var',
  'stage4_unit2_conv2_weight'
]

end = [
  'bn1_beta',
  'bn1_gamma',
  'bn1_moving_mean',
  'bn1_moving_var',
  'fc1_weight',
  'fc1_bias'
]


# create placeholder in batch
holders, values = list(), list()
names = before + stage1_unit1 + stage1_unit2 + \
        stage2_unit1 + stage2_unit2 + \
        stage3_unit1 + stage3_unit2 + \
        stage4_unit1 + stage4_unit2 + end

# run and calculate test accuracy
qtype1 = hcl.Fixed(16, 14)
qtype2 = hcl.Fixed(16, 14)
correct_sum, correct_top5 = 0, 0

params = arg_params.copy()
params.update(aux_params)
for name in names:
    val = params[name].asnumpy()
    ph  = hcl.placeholder(val.shape, name)
    holders.append(ph)
    values.append(hcl.asarray(val, dtype=hcl.Float()))

# build the function
input_image = hcl.placeholder((batch_size, 3, 224, 224), "input_image")
resnet = hcl.placeholder((batch_size, 1000), "resnet")

# create scheme and build 
arg_list = [input_image, resnet] + holders
scheme = hcl.create_scheme(arg_list, build_resnet)
s = hcl.create_schedule_from_scheme(scheme)
a = s.to(build_resnet.s2_u1_add, target.xcel)
b = s.to(build_resnet.s3_u1_add, target.host)
print(a, type(a))
# print(hcl.lower(s))
s.dataflow_graph(plot=True)
f = hcl.build(s, target=target)

# ---------------------------
# result validation
# ---------------------------
images = np.load('images.npy')
labels = np.load('labels.npy')
with open('synset.txt', 'r') as fin:
    tags = [l.rstrip() for l in fin]

for i in range(len(images)):
    input_image_np = np.expand_dims(images[i], axis=0)
    label = labels[i]
    input_image_hcl = hcl.asarray(input_image_np, dtype=hcl.Float())
    output_hcl = hcl.asarray(np.zeros((batch_size, 1000)))

    # prediction from hcl model
    f(input_image_hcl, output_hcl, *values)
    prediction = np.argmax(output_hcl.asnumpy(), axis=1)
    x = output_hcl.asnumpy()[0]
    top5 = np.argsort(x)[-5:] 
    print(prediction, label, i, tags[int(label)])
    correct_sum += np.sum(np.equal(prediction, label))
    if label in top5:
        correct_top5 += np.sum(np.equal(1, 1))

print(str(qtype1) + ", " + str(qtype2) + \
": Accuracy over 50 test images is: {}".format(correct_sum / 50.))
print(str(qtype1) + ", " + str(qtype2) + \
": Accuracy over 50 test images is: {}".format(correct_top5 / 50.))
