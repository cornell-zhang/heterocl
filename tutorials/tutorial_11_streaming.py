"""
Use the Data Placement and Streaming Feature
=======================
**Author**: Shaojie

In this tutorial, we show how to use the .to() APIs to create
on-chip and off-chip data streaming channels.
"""
import os
import numpy as np
import heterocl as hcl

##############################################################################
# Create Kernel Function
# -------------------
# Create a program with HeteroCL, and apply schedule primitives to optimize 
# the performance. Here we use a simple vector-add and vector-mul kernel:
n = 64
dtype = hcl.Int(32)

vector_1 = hcl.placeholder((n,), dtype=dtype)
vector_2 = hcl.placeholder((n,), dtype=dtype)

def kernel(vector_1, vector_2):
    add = hcl.compute((n,), lambda x: vector_1[x] + vector_2[x], "add")
    mul = hcl.compute((n,), lambda x: add[x] * 2, "mul")
    return mul

s = hcl.create_schedule([vector_1, vector_2], kernel)

##############################################################################
# Create Custom Platforms
# -----------------------
# HeteroCL has many built-in platforms (including aws_f1, zc706, stratix10, e.t.c). 

config = {
    "host" : hcl.dev.cpu("intel", "e5"),
    "xcel" : [
        hcl.dev.fpga("xilinx", "xcu250")
    ]
}

mode = "debug" if os.system("which v++ >> /dev/null") != 0 else "hw_exe"
target = hcl.platform.custom(config)
target.config(compile="vitis", mode="hw_exe")

s.to([vector_1, vector_2], target.xcel, mode=hcl.IO.Stream)
s.to(kernel.mul, target.host, mode=hcl.IO.Stream)
s.to(kernel.add, s[kernel.mul])

##############################################################################
# Run the Compilation
# --------------------
# Run the compilation: after you are done with optimization and compute offloading, 
# you will get a compiled function from HeteroCL, and this function to can be used to 
# process the input array passed from HeteroCL python interface. The input data will 
# be passed to the accelerator function, and all the computations offloaded to FPGA i
# will be executed by the HLS tools running under the hood. 

if mode == "debug":
    print(hcl.build(s, target))

else:
    hcl_m1 = hcl.asarray(np.random.randint(10, size=(n,)), dtype=dtype)
    hcl_m2 = hcl.asarray(np.random.randint(10, size=(n,)), dtype=dtype)
    hcl_m3 = hcl.asarray(np.zeros((n,)), dtype=dtype)
    f = hcl.build(s, target)
    f(hcl_m1, hcl_m2, hcl_m3)
    
