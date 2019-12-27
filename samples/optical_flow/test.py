import heterocl as hcl
import hlib
import numpy as np
import os, sys

batch_size = 1
hcl.init(hcl.UInt(32))
dtype = hcl.UInt(32)

# setup target using vivado 
tool = hcl.tool.vivado("csim")
target = hcl.platform.zc706

def test(target=target):
    image = hcl.placeholder((10,10), "input_image")
    output = hcl.placeholder((10,10), "output_image")

    def kernel(input_image, output):

        @hcl.def_([(10,10), (10,10)])
        def test_func(input_image, output):
            """ use same buffer if name matches"""
            s = hcl.scalar(1)
            output = hcl.compute((10, 10), 
                lambda x, y: input_image[x, y] * 2, name="output")
            # interm = hcl.compute((10, 10), lambda x, y: 3, name="sw")
            hcl.update(output, lambda x, y: 3 + output[x,y])
            

        @hcl.def_([(10,10)])
        def add(output):
            return hcl.compute((10, 10), 
                lambda x, y: output[x, y] + 10, name="output")

        test_func(input_image, output)
        return add(output)

    s = hcl.create_schedule([image, output], kernel)

    print(hcl.lower(s))
    # print(kernel.test_func.output_x)
    return hcl.build(s, target)

hcl_input  = hcl.asarray(np.zeros((10,10)), dtype)    
hcl_output = hcl.asarray(np.zeros((10,10)), dtype)    

f = test("llvm")
f(hcl_input, hcl_output)
