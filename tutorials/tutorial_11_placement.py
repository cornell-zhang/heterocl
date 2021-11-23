"""
Use the Data Placement and Streaming Feature
=======================
**Author**: Hecmay
In this tutorial, we show how to use the .to() APIs to offload specific 
compute regions to FPGA.
"""
import os
import numpy as np
import heterocl as hcl

##############################################################################
# Create Kernel Function
# -------------------
# Create a program with HeteroCL, and apply schedule primitives to optimize 
# the performance. Here we use the GEMM as an example:
m = 64
k = 64
n = 64
dtype = hcl.Int(32)

matrix_1 = hcl.placeholder((m, k), dtype=dtype)
matrix_2 = hcl.placeholder((k, n), dtype=dtype)

def kernel(matrix_1, matrix_2):
    r = hcl.reduce_axis(0, k, 'k')
    return hcl.compute((m, n),
            lambda x, y: hcl.sum(matrix_1[x, r] * matrix_2[r, y],
                                 axis=r, dtype=dtype),
            dtype=dtype,
            name="out_matrix")

s = hcl.create_schedule([matrix_1, matrix_2], kernel)

##############################################################################
# Move tensor to device
# -----------------------
# HeteroCL has many built-in Platforms (including aws_f1, zc706, stratix 10, e.t.c). 
# Here we use the zc706 Platform as an example. The zc706 Platform is associated with 
# Vivado HLS tool by default. By using .to() primitives, tensors will be moved into device 
# scope, and all computations depending on these tensors will be performed on FPGA. 
# Note that you also need to move the result tensor back to host. 

target = hcl.Platform.xilinx_zc706
s.to([matrix_1, matrix_2], target.xcel)
s.to(kernel.out_matrix, target.host)

##############################################################################
# Run the Compilation
# --------------------
# Run the compilation: after you are done with optimization and compute offloading, 
# you will get a compiled function from HeteroCL, and this function to can be used to 
# process the input array passed from HeteroCL python interface. The input data will 
# be passed to the accelerator function, and all the computations offloaded to FPGA
# will be executed by the HLS tools running under the hood. 

if os.getenv("LOCAL_CI_TEST"):
    target.config(compiler="vivado_hls", mode="csyn")
    f = hcl.build(s, target)

    hcl_m1 = hcl.asarray(np.random.randint(10, size=(m, k)), dtype=dtype)
    hcl_m2 = hcl.asarray(np.random.randint(10, size=(k, n)), dtype=dtype)
    hcl_m3 = hcl.asarray(np.zeros((m, n)), dtype=dtype)
    f(hcl_m1, hcl_m2, hcl_m3)

else:
    target.config(compiler="vivado_hls", mode="debug")
    print(hcl.build(s, target))

