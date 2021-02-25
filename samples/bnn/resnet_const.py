import heterocl as hcl
import hlib.op.bnn as bnn
import hlib.op.nn_ as nn
import numpy as np
from functools import reduce
import os, time, sys, argparse
import torch
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--pytorch', type=bool, default=False,
                    help="Use PyTorch's dataloader? (default: True)")
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
    target = hcl.Platform.zc706
    if args.vitis:
        print("Use Vitis to compile")
        target.config(compile="vitis", mode="hw_exe", project="project-vitis")
    else:
        target.config(compile="vivado_hls", mode="csim|csyn|cosim")

def RSign(data, alpha, name="rsign", dtype=hcl.UInt(1)):
    assert data.shape[1] == alpha.shape[0]
    return hcl.compute(data.shape, lambda nn, cc, ww, hh: 
                        hcl.select(data[nn,cc,ww,hh] + alpha[cc] > 0, 1, 0),
                        name=name, dtype=dtype)

def packed_RSign(data, alpha, name="rsign"): # useless dtype here
    assert data.shape[1] == alpha.shape[0]
    batch, channel, out_height, out_width = data.shape
    bitwidth = channel if channel <= 32 else 32 # pack channels
    def genpack(nn, cc, hh, ww):
        out = hcl.scalar(0, name=name+"_pack", dtype=hcl.UInt(bitwidth))
        with hcl.for_(0, bitwidth) as k:
            out[0][(k+1) : k] = hcl.select(data[nn, cc*bitwidth+k, hh, ww] + alpha[cc*bitwidth+k] > 0, 1, 0)
        return out[0]
    return hcl.compute((batch, channel//bitwidth, out_height, out_width),
                        genpack, name=name, dtype=hcl.UInt(bitwidth))

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
        self.params["rprelu1"] = [hcl.const_tensor(params[i],"w_{}_rprelu1_{}".format(name,i),qtype_float) for i in range(3)]
        self.params["rprelu2"] = [hcl.const_tensor(params[i],"w_{}_rprelu2_{}".format(name,i),qtype_float) for i in range(3,6)]
        self.params["rsign1"] = hcl.const_tensor(params[6],"w_{}_rsign1".format(name),qtype_float)
        self.params["rsign2"] = hcl.const_tensor(params[7],"w_{}_rsign2".format(name),qtype_float)
        if name.split("layer")[1][0] == "1":
            bitwidth = 16
        elif name.split("layer")[1][0] == "2":
            bitwidth = 32
        else:
            bitwidth = 32 # do NOT use 64 bitwidth!
        dtype = hcl.UInt(bitwidth) # remember to set
        self.params["conv1"] = hcl.const_tensor(params[8],"w_{}_conv1".format(name),hcl.UInt(bitwidth))
        self.params["bn1"] = [hcl.const_tensor(params[i],"w_{}_bn1_{}".format(name,i),qtype_float) for i in range(9,13)]
        self.params["conv2"] = hcl.const_tensor(params[13],"w_{}_conv2".format(name),hcl.UInt(bitwidth))
        self.params["bn2"] = [hcl.const_tensor(params[i],"w_{}_bn2_{}".format(name,i),qtype_float) for i in range(14,18)]
        self.stride = stride
        self.flag = in_planes != planes
        self.name = name

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # 1st residual block
        rsign1 = packed_RSign(x, self.params["rsign1"], name=self.name+"_rsign1")
        conv1 = bnn.packed_conv2d_nchw(rsign1, self.params["conv1"], padding=[1,1], strides=[self.stride,self.stride], name=self.name+"_conv1", out_dtype=qtype_int, mac=False) # no bias!
        bn1, _, _ = nn.batch_norm(conv1, *self.params["bn1"], name=self.name+"_bn1",dtype=qtype_float)
        if self.stride != 1 or self.flag:
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
        rsign2 = packed_RSign(rprelu1, self.params["rsign2"], name=self.name+"_rsign2")
        conv2 = bnn.packed_conv2d_nchw(rsign2, self.params["conv2"], strides=[1,1], padding=[1,1], name=self.name+"_conv2",out_dtype=qtype_int, mac=False)
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
        self.params["conv1"] = hcl.const_tensor(params[0],"w_conv1",qtype_float)
        self.params["bn1"] = [hcl.const_tensor(params[i],"w_bn1_{}".format(i),qtype_float) for i in range(1,5)]
        self.params["layer1"] = params[5:59]
        self.params["layer2"] = params[59:113]
        self.params["layer3"] = params[113:167]
        self.params["linear"] = [hcl.const_tensor(params[i],"w_fc_{}".format(i),qtype_float) for i in range(167,169)]
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
        conv1 = nn.conv2d_nchw(x, self.params["conv1"], strides=[1, 1], padding=[1, 1], name="conv0", out_dtype=qtype_float)
        bn, _, _ = nn.batch_norm(conv1, *self.params["bn1"], name="bn1",dtype=qtype_float)
        layer1 = self.layer1(bn)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        kernel_size = layer3.shape[3]
        avgpool = nn.avg_pool2d_LB(layer3, pooling=[kernel_size, kernel_size], stride=[kernel_size, kernel_size], padding=[0, 0], name="avgpool", dtype=qtype_float)
        flat = nn.flatten(avgpool, name="flatten", dtype=qtype_float)
        out = nn.dense(flat, self.params["linear"][0], bias=self.params["linear"][1], name="fc", out_dtype=qtype_float)
        return out

def build_resnet20(input_image): # params are placeholders here
    resnet = ResNet(BasicBlock, [3, 3, 3], hcl_params)
    return resnet(input_image)

def build_resnet20_inf(target=target):

    if isinstance(target,hcl.Platform):
        s.to([input_image], target.xcel)
        s.to(build_resnet20.fc, target.host)

    return hcl.build(s, target=target)

def build_resnet20_stream_inf(target=target):

    layer_names = list(build_resnet20.__dict__.keys())
    new_layer = []
    for layer in layer_names:
        if not ("LB" in layer or "w_" in layer):
            new_layer.append(layer)
    layer_names = new_layer
    if not args.stream and not args.vitis:
        s.partition(input_image,dim=4)
    for layer in layer_names:
        s_layer = getattr(build_resnet20,layer)
        if "pad" in layer:
            s[s_layer].pipeline(s_layer.axis[2])
            if not args.stream:
                s.partition(s_layer,dim=4) # avoid using with streaming
        elif "bn" in layer:
            s[s_layer].pipeline(s_layer.axis[3])
        elif "rsign" in layer or "residual" in layer or "rprelu" in layer:
            s[s_layer].pipeline(s_layer.axis[3])
            if not args.stream:
                s.partition(s_layer,dim=4)
        elif "conv" in layer and "pad" not in layer:
            s[s_layer].pipeline(s_layer.axis[3])
            s[s_layer].reorder(s_layer.axis[2],s_layer.axis[3],s_layer.axis[1])
            if layer == "layer2_0_conv1" or layer == "layer3_0_conv1":
                continue # stride=2
            s_pad = getattr(build_resnet20,layer+"_pad")
            LB = s.reuse_at(s_pad._op,s[s_layer],s_layer.axis[2],layer+"_LB")
            WB = s.reuse_at(LB,s[s_layer],s_layer.axis[3],layer+"_WB")
        elif "concat" in layer:
            s[s_layer].pipeline(s_layer.axis[2])
        elif "avgpool" in layer:
            if layer == "avgpool":
                s[s_layer].pipeline(s_layer.axis[1]) # (hh,ww) = (1,1)
            else:
                s[s_layer].pipeline(s_layer.axis[2])
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
        straight_layer_names.remove("layer2_0_avgpool_res")
        straight_layer_names.remove("layer3_0_avgpool_res")
        straight_layer_names.remove("avgpool_res")
        import heterocl.tvm as tvm
        for i,layer in enumerate(straight_layer_names):
            if i == len(straight_layer_names) - 1:
                break
            if "avgpool" not in layer:
                layer1 = getattr(build_resnet20,layer)
                layer2 = getattr(build_resnet20,list(straight_layer_names)[i+1])
                shape = layer1._op.shape
            else:
                layer1 = getattr(getattr(build_resnet20,layer),layer+"_res")
                layer2 = getattr(build_resnet20,list(straight_layer_names)[i+1])
                shape = getattr(build_resnet20,layer+"_res")._op.shape
            depth = tvm.ir_pass.Simplify(reduce(lambda x, y: x * y, shape))
            s.to(layer1, layer2, depth=depth.value)
        # residual streaming
        f = build_resnet20
        for layer in range(1,4):
            for bb in range(3):
                if bb == 0:
                    prev = "bn1" if layer == 1 else "layer{}_2_rprelu2".format(layer-1)
                else:
                    prev = "layer{}_{}_rprelu2".format(layer,bb-1)
                if not (layer != 1 and bb == 0):
                    shape = getattr(f,prev)._op.shape
                    depth = tvm.ir_pass.Simplify(reduce(lambda x, y: x * y, shape))
                    s.to(getattr(f,prev),
                        getattr(f,"layer{}_{}_residual1".format(layer,bb)),
                        depth=depth.value)
                else: # 2_0 3_0
                    shape = getattr(f,prev)._op.shape
                    depth = tvm.ir_pass.Simplify(reduce(lambda x, y: x * y, shape))
                    s.to(getattr(f,prev),
                         getattr(f,"layer{}_{}_avgpool".format(layer,bb)),
                         depth=depth.value)
                    avgpool_s = getattr(getattr(f,"layer{}_{}_avgpool".format(layer,bb)),"layer{}_{}_avgpool_res".format(layer,bb))
                    shape = getattr(f,"layer{}_{}_avgpool_res".format(layer,bb))._op.shape
                    depth = tvm.ir_pass.Simplify(reduce(lambda x, y: x * y, shape))
                    s.to(avgpool_s,
                         getattr(f,"layer{}_{}_concat".format(layer,bb)),
                         depth=depth.value)
                    shape = getattr(f,"layer{}_{}_concat".format(layer,bb))._op.shape
                    depth = tvm.ir_pass.Simplify(reduce(lambda x, y: x * y, shape))
                    s.to(getattr(f,"layer{}_{}_concat".format(layer,bb)),
                         getattr(f,"layer{}_{}_residual1".format(layer,bb)),
                         depth=depth.value)
                shape = getattr(f,"layer{}_{}_rprelu1".format(layer,bb))._op.shape
                depth = tvm.ir_pass.Simplify(reduce(lambda x, y: x * y, shape))
                s.to(getattr(f,"layer{}_{}_rprelu1".format(layer,bb)),
                     getattr(f,"layer{}_{}_residual2".format(layer,bb)),
                     depth=depth.value)

    if isinstance(target,hcl.Platform):
        s.to([input_image], target.xcel)
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
hcl_params = []
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
        # bitpacking
        if temp.shape[1] == 1: # channel
            np_type = np.bool
        elif temp.shape[1] == 16:
            np_type = np.uint16
        elif temp.shape[1] == 32:
            np_type = np.uint32
        elif temp.shape[1] == 64:
            np_type = np.uint32
        arr = temp.transpose(0,2,3,1)
        arr = np.packbits(arr.astype(np.bool),
                axis=3,bitorder="little").view(np_type)
        new_params[new_key] = arr.transpose(0,3,1,2)
    else:
        new_params[new_key] = np.array(params[key])
    hcl_params.append(new_params[new_key])
params = new_params

input_image = hcl.placeholder((batch_size,3,32,32),"input_image",dtype=qtype_float)
s = hcl.create_schedule([input_image], build_resnet20)

hcl_out = hcl.asarray(np.zeros((batch_size,10)).astype(np.float),dtype=qtype_float)

if __name__ == "__main__":
    resnet20 = build_resnet20_inf()
    print("Finish building function.")

    correct_sum = 0
    for i, (images, labels) in enumerate(test_loader):
        images = np.array(images)
        labels = np.array(labels)
        hcl_image = hcl.asarray(images, dtype=qtype_float)
        resnet20(hcl_image, hcl_out)
        prediction = np.argmax(hcl_out.asnumpy(), axis=1)
        correct_sum += np.sum(np.equal(prediction, labels))
        if (i+1) % 10 == 0:
            print("Done {} batches.".format(i+1))
        if (i+1) * batch_size == test_size:
            break
    print("Testing accuracy: {}".format(correct_sum / float(test_size)))
