import heterocl as hcl
import hlib.op.bnn as bnn
import hlib.op.nn as nn
import numpy as np
import os, time, sys, argparse
import torch
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--pytorch', type=bool, default=False,
                    help="Use PyTorch's dataloader? (default: False)")
parser.add_argument('--vitis', type=bool, default=False,
                    help='Use Vitis to compile? (default: False)')
parser.add_argument('--opt', type=bool, default=False,
                    help='Use optimization? (default: False)')
parser.add_argument('--stream', type=bool, default=False,
                    help='Use data streaming? (default: False)')
args = parser.parse_args()

test_size = 100
qtype_bit = hcl.UInt(1) # weights
qtype_int = hcl.Int(8)
if __name__ == "__main__":
    batch_size = 10
    qtype_float = hcl.Fixed(24,12)
    target = None
else: # vhls
    batch_size = 1
    qtype_float = hcl.Fixed(32,12) # for interface synthesis
    target = hcl.platform.zc706
    if args.vitis:
        print("Use Vitis to compile")
        target.config(compile="vitis", mode="hw_exe")
    else:
        target.config(compile="vivado_hls", mode="csyn")
# qtype_packed = hcl.UInt(32)

def RSign(data, alpha, name="rsign"):
    assert data.shape[1] == alpha.shape[0]
    return hcl.compute(data.shape, lambda nn, cc, ww, hh: 
                        hcl.select(data[nn,cc,ww,hh] + alpha[cc] > 0, 1, 0),
                        name=name, dtype=hcl.UInt(1))

def RPReLU(data, x0, y0, beta, name="rprelu", dtype=None):
    assert data.shape[1] == beta.shape[0] \
        and x0.shape[0] == y0.shape[0] \
        and beta.shape[0] == x0.shape[0]
    dtype = data.dtype if dtype == None else dtype
    return hcl.compute(data.shape, lambda nn, cc, ww, hh:
                        hcl.select(data[nn,cc,ww,hh] + x0[cc] > 0,
                        data[nn,cc,ww,hh] + x0[cc],
                        beta[cc] * (data[nn,cc,ww,hh] + x0[cc])) + y0[cc],
                        name=name, dtype=dtype)

class BasicBlock():

    def __init__(self, in_planes, planes, stride, params, name="bb"):
        self.params = dict()
        self.params["rprelu1"] = params[:3]
        self.params["rprelu2"] = params[3:6]
        self.params["rsign1"] = params[6]
        self.params["rsign2"] = params[7]
        self.params["conv1"] = params[8]
        self.params["bn1"] = params[9:13]
        self.params["conv2"] = params[13]
        self.params["bn2"] = params[14:18]
        self.stride = stride
        self.flag = in_planes != planes
        self.name = name

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # 1st residual block
        rsign1 = RSign(x, self.params["rsign1"], name=self.name+"_rsign1")
        conv1 = bnn.conv2d_nchw(rsign1, self.params["conv1"], padding=[1,1], strides=[self.stride,self.stride], name=self.name+"_conv1", out_dtype=qtype_int) # no bias!
        bn1, _, _ = nn.batch_norm(conv1, *self.params["bn1"], name=self.name+"_bn1",dtype=qtype_float)
        if self.stride != 1 or self.flag:
            # avgpool = nn.avg_pool2d_nchw(x, pooling=[2,2],
            #                              stride=[2,2], padding=[0,0],
            #                              name=self.name+"_avgpool",dtype=qtype_float)
            avgpool = nn.avg_pool2d_LB(x, pooling=[2,2],
                                       stride=[2,2], padding=[0,0],
                                       name=self.name+"_avgpool",dtype=qtype_float)
            # dont use nn.concatenate!
            shape = avgpool.shape
            shortcut = hcl.compute((shape[0], shape[1]*2, shape[2], shape[3]),
                                    lambda nn, cc, ww, hh: avgpool[nn, cc % shape[1], ww, hh],
                                    name=self.name+"_concat",dtype=qtype_float)
        else:
            shortcut = x
        residual1 = hcl.compute(bn1.shape, lambda nn, cc, ww, hh:
                                bn1[nn, cc, ww, hh] + shortcut[nn, cc, ww, hh],
                                name=self.name+"_residual1",dtype=qtype_float)
        # 2nd residual block
        rprelu1 = RPReLU(residual1, *self.params["rprelu1"], name=self.name+"_rprelu1",dtype=qtype_float)
        rsign2 = RSign(rprelu1, self.params["rsign2"], name=self.name+"_rsign2")
        conv2 = bnn.conv2d_nchw(rsign2, self.params["conv2"], strides=[1,1], padding=[1,1], name=self.name+"_conv2",out_dtype=qtype_int)
        bn2, _, _ = nn.batch_norm(conv2, *self.params["bn2"], name=self.name+"_bn2",dtype=qtype_float)
        residual2 = hcl.compute(rprelu1.shape, lambda nn, cc, ww, hh:
                                bn2[nn, cc, ww, hh] + rprelu1[nn, cc, ww, hh],
                                name=self.name+"_residual2",dtype=qtype_float)
        rprelu2 = RPReLU(residual2, *self.params["rprelu2"], name=self.name+"_rprelu2",dtype=qtype_float)
        return rprelu2

class Sequential():

    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

class ResNet():

    def __init__(self, block, num_blocks, params):
        self.in_planes = 16
        self.params = dict()
        self.params["conv1"] = params[0]
        self.params["bn1"] = params[1:5]
        self.params["layer1"] = params[5:59]
        self.params["layer2"] = params[59:113]
        self.params["layer3"] = params[113:167]
        self.params["linear"] = params[167:]
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, params=self.params["layer1"], id=0)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, params=self.params["layer2"], id=1)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, params=self.params["layer3"], id=2)

    def _make_layer(self, block, planes, num_blocks, stride, params, id):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, params[i*18:(i+1)*18], name="layer{}_{}".format(id+1,i)))
            self.in_planes = planes

        return Sequential(*layers)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        conv1 = nn.conv2d_nchw(x, self.params["conv1"], strides=[1, 1], padding=[1, 1], name="conv1", out_dtype=qtype_float)
        bn, _, _ = nn.batch_norm(conv1, *self.params["bn1"], name="bn1",dtype=qtype_float)
        layer1 = self.layer1(bn)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        kernel_size = layer3.shape[3]
        # avgpool = nn.avg_pool2d_nchw(layer3, pooling=[kernel_size, kernel_size], stride=[kernel_size, kernel_size], padding=[0, 0], name="avgpool", dtype=qtype_float)
        avgpool = nn.avg_pool2d_LB(layer3, pooling=[kernel_size, kernel_size], stride=[kernel_size, kernel_size], padding=[0, 0], name="avgpool", dtype=qtype_float)
        flat = nn.flatten(avgpool, name="flatten", dtype=qtype_float)
        out = nn.dense(flat, self.params["linear"][0], bias=self.params["linear"][1], name="fc", out_dtype=qtype_float)
        return out

def build_resnet20(*params): # params are placeholders here
    resnet = ResNet(BasicBlock, [3, 3, 3], params[1:])
    return resnet(params[0])

def build_resnet20_inf(params, target=target):

    if isinstance(target,hcl.platform):
        s.to([input_image] + hcl_ph, target.xcel)
        s.to(build_resnet20.fc, target.host)

    return hcl.build(s, target=target)

def build_resnet20_stream_inf(params, target=target):

    layer_names = list(build_resnet20.__dict__.keys())
    layer_names.remove("layer2_0_avgpool_LB")
    layer_names.remove("layer3_0_avgpool_LB")
    layer_names.remove("avgpool_LB")
    for layer in layer_names:
        s_layer = getattr(build_resnet20,layer)
        if "pad" in layer:
            s[s_layer].pipeline(s_layer.axis[2])
            # s.partition(s_layer,dim=4) # avoid using with streaming
        elif "bn" in layer:
            s[s_layer].pipeline(s_layer.axis[3])
        elif "rsign" in layer or "residual" in layer or "rprelu" in layer:
            s[s_layer].pipeline(s_layer.axis[3])
        elif "conv" in layer and "pad" not in layer:
            if "layer2_0" in layer or "layer3_0" in layer:
                continue # stride=2
            s[s_layer].pipeline(s_layer.axis[3])
            s_pad = getattr(build_resnet20,layer+"_pad")
            LB = s.reuse_at(s_pad._op,s[s_layer],s_layer.axis[2],layer+"_LB")
            WB = s.reuse_at(LB,s[s_layer],s_layer.axis[3],layer+"_WB")
            # s.partition(ph_dict[layer+"_weight"],hcl.Partition.Cyclic,factor=3,dim=4) # avoid too many registers
            # s.partition(s_layer,dim=2) # avoid using with streaming
        elif "avgpool" in layer:
            s[s_layer].pipeline(s_layer.axis[2])
            # s.partition(s_layer,dim=4) # avoid using with streaming
        elif "flatten" in layer:
            s[s_layer].pipeline(s_layer.axis[1])
        elif "fc_matmul" in layer:
            s[s_layer].pipeline(s_layer.axis[2])
            s_fc = getattr(build_resnet20,"fc")
            s[s_fc].pipeline(s_fc.axis[1])

    # streaming across layers (straight line)
    if args.stream:
        straight_layer_names = layer_names.copy()
        straight_layer_names.remove("layer2_0_avgpool")
        straight_layer_names.remove("layer2_0_concat")
        straight_layer_names.remove("layer3_0_avgpool")
        straight_layer_names.remove("layer3_0_concat")
        for i,layer in enumerate(straight_layer_names):
            if i == len(straight_layer_names) - 1:
                break
            layer1 = getattr(build_resnet20,layer)
            layer2 = getattr(build_resnet20,list(straight_layer_names)[i+1])
            s.to(layer1,s[layer2])
        # residual streaming
        f = build_resnet20
        for layer in range(1,4):
            for bb in range(3):
                if bb == 0:
                    prev = "bn1" if layer == 1 else "layer{}_2_rprelu2".format(layer-1)
                else:
                    prev = "layer{}_{}_rprelu2".format(layer,bb-1)
                if not (layer != 1 and bb == 0):
                    s.to(getattr(f,prev),
                        getattr(f,"layer{}_{}_residual1".format(layer,bb)))
                else: # 2_0 3_0
                    s.to(getattr(f,prev),
                        getattr(f,"layer{}_{}_avgpool".format(layer,bb)))
                    s.to(getattr(f,"layer{}_{}_avgpool".format(layer,bb)),
                        getattr(f,"layer{}_{}_concat".format(layer,bb)))
                    s.to(getattr(f,"layer{}_{}_concat".format(layer,bb)),
                        getattr(f,"layer{}_{}_residual1".format(layer,bb)))
                s.to(getattr(f,"layer{}_{}_rprelu1".format(layer,bb)),
                    getattr(f,"layer{}_{}_residual2".format(layer,bb)))

    if isinstance(target,hcl.platform):
        s.to([input_image] + hcl_ph, target.xcel)
        s.to(build_resnet20.fc, target.host)

    return hcl.build(s, target=target)

def load_cifar10():
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    normalize
        ])
    test_set = torchvision.datasets.CIFAR10(root='.', train=False,
                                           download=True, transform=transform_test)
    if not args.pytorch:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=2)
    else:
        print("NOT using Pytorch's dataloader")
        images = ((test_set.data * 1.0 / 255 - mean) / std).transpose((0,3,1,2))
        labels = np.array(test_set.targets)
        images = np.split(images, images.shape[0] // batch_size)
        labels = np.split(labels, labels.shape[0] // batch_size)
        test_loader = zip(images, labels)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return test_loader

params = torch.load("pretrained-models/model_react-resnet20-8bitBN.pt", map_location=torch.device("cpu"))
print("Loading the data.")
test_loader = load_cifar10()
new_params = dict()
for key in params:
    params[key] = params[key].numpy()
    new_key = key.replace(".","_")
    if "num_batches_tracked" in key:
        continue
    elif "rprelu" in key or "binarize" in key:
        new_params[new_key] = np.array(params[key]).reshape(-1)
    elif "conv" in key and "layer" in key:
        temp = np.sign(params[key])
        temp[temp < 0] = 0 # change from {-1,1} to {0,1}
        new_params[new_key] = temp
    else:
        new_params[new_key] = np.array(params[key])
params = new_params

hcl_array = []
for name in params:
    dtype = qtype_bit if "conv" in name and "layer" in name else qtype_float
    hcl_array.append(hcl.asarray(params[name],dtype=dtype))
hcl_out = hcl.asarray(np.zeros((batch_size,10)).astype(np.float),dtype=qtype_float)

hcl_ph = []
ph_dict = {}
input_image = hcl.placeholder((batch_size,3,32,32),"input_image",dtype=qtype_float)
for name in params:
    dtype = qtype_bit if "conv" in name and "layer" in name else qtype_float
    hcl_ph.append(hcl.placeholder(params[name].shape,name,dtype=dtype))
    ph_dict[name] = hcl_ph[-1]

s = hcl.create_schedule([input_image] + hcl_ph, build_resnet20)

if __name__ == "__main__":
    resnet20 = build_resnet20_inf(params)
    print("Finish building function.")

    correct_sum = 0
    for i, (images, labels) in enumerate(test_loader):
        images = np.array(images)
        labels = np.array(labels)
        hcl_image = hcl.asarray(images, dtype=qtype_float)
        resnet20(hcl_image, *hcl_array, hcl_out)
        prediction = np.argmax(hcl_out.asnumpy(), axis=1)
        correct_sum += np.sum(np.equal(prediction, labels))
        if (i+1) % 10 == 0:
            print("Done {} batches.".format(i+1))
        if (i+1) * batch_size == test_size:
            break
    print("Testing accuracy: {}".format(correct_sum / float(test_size)))