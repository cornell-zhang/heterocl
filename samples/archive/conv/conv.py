import heterocl as hcl
import hlib
import numpy as np
from PIL import Image
from urllib.request import urlopen

batch_size = 1
hcl.init(hcl.UInt(32))
dtype = hcl.UInt(32)
image_size = ()
kernel_size = 3

# setup target using vivado 
tool = hcl.tool.vivado("csim")
target = hcl.Platform.xilinx_zc706

def conv():
    image = hcl.placeholder((batch_size, 1, 256, 256), "input_image")
    k1 = hcl.placeholder((1, 1, 3, 3), "kernel_1")
    k2 = hcl.placeholder((1, 1, 3, 3), "kernel_2")

    def kernel(input_image, kernel_1, kernel_2):

        # return tensor required (cannot do def_())
        interm_shape = (1,1,254,254)
        output_shape = (1,1,252,252)

        # make compute wrapped in hcl def
        module1 = hcl.def_([input_image.shape, kernel_1.shape, interm_shape], name="conv1")(hlib.nn.conv2d_nchw_imp)
        module2 = hcl.def_([interm_shape, kernel_2.shape, output_shape], name="conv2")(hlib.nn.conv2d_nchw_imp)
        conv1 = hcl.compute(interm_shape, lambda *args: 0)  
        conv2 = hcl.compute(output_shape, lambda *args: 0)  
        module1(input_image, kernel_1, conv1)
        module2(conv1, kernel_2, conv2)

        # derivative module for normalization 
        return hcl.compute(output_shape, lambda *args: conv2[args], name="derv")

    s = hcl.create_schedule([image, k1, k2], kernel)

    # data moved to local  
    i0, k10, k20 = s.to([image, k1, k2], target.fpga)
    # s.to([i0, k10], s[kernel.conv1])
    # s.to([k20], s[kernel.conv2])
    s.to(kernel.derv, target.cpu)

    # create stream channel between modules 
    print(type(target.fpga), hcl.lower(s))
    return hcl.build(s, target)

# Load sample data
img = Image.open(urlopen('http://i.stack.imgur.com/8zINU.gif'))
kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel_y = np.flip(kernel_x.T.T, axis=0)
img = np.array(img)

img = img[np.newaxis, ...]
img = img[np.newaxis, ...]
kernel_x = kernel_x[np.newaxis, ...]
kernel_x = kernel_x[np.newaxis, ...]
kernel_y = kernel_y[np.newaxis, ...]
kernel_y = kernel_y[np.newaxis, ...]

hcl_input  = hcl.asarray(img, dtype)    
kernel_x   = hcl.asarray(kernel_x, dtype)
kernel_y   = hcl.asarray(kernel_y, dtype)
hcl_output = hcl.asarray(np.zeros((1,1,254,254)), dtype)    

f = conv()
f(hcl_input, kernel_x, kernel_y, hcl_output)
