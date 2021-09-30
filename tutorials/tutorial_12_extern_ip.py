"""
Insert HLS IP into HeteroCL program
=======================
**Author**: Hecmay
In this tutorial, we show how to insert manually written HLS IP into HCL program
"""

import heterocl as hcl
import numpy as np
import numpy.testing as tst
import os

from hlib.op.extern import (
    create_extern_module, register_extern_ip, 
    register_tensors, include_dependency)

##############################################################################
# Define HLS IP 
# -------------------
# Define the HLS code that contains the function to be included in the HCL program. Here we use a vector add as an example. 
vadd_hls_ip = """
void vadd(const float* in1, float* in2, int size) {
    for (int j = 0; j < size; j++) {
#pragma HLS PIPELINE II=1
        in2[j] = in1[j] + 1;
    }
}
"""
with open("vadd.cpp", "w") as fp:
    fp.write(vadd_hls_ip)

##############################################################################
# Register HLS IP 
# -------------------
# Register the external HLS IP function in HCL. The registered HLS function will be used later to 
@register_extern_ip(type="vhls")
def vadd_vhls_ip(op1, op2, size, name=None):
    # define HLS function name
    if name is None: name = "vadd"
    # define a HCL stage and registers the input and output 
    # tensors that 
    with hcl.Stage("ExternModule.vadd") as Module:
        register_tensors([op1, op2])
    # define the function name and input arguments
    Module.ext_ip_name = name
    Module.inputs = [op1, op2, size]
    # define the source code and dependency files (e.g., headers) 
    deps = os.path.dirname(os.path.abspath(__file__))
    source = [ "vadd.cpp" ]
    Module.source = include_dependency(source)
    create_extern_module(Module, ip_type="HLS")

##############################################################################
# Define HCL program using external HLS function
# -----------------------
# We first define the placeholders as the inputs to our program. In the main function, we first call HLS IP function and then use imperative coding style to manipulate data returned from HLS IP.
dtype = hcl.Float(32)
hcl.init(dtype)
size = 1024
op1 = hcl.placeholder((size,), dtype=dtype, name="op1")
op2 = hcl.placeholder((size,), dtype=dtype, name="op2")
# call external HLS function inside HCL program
def math_func(op1, op2):
    vadd_vhls_ip(op1, op2, size)
    op2[0] += 1

# host-accelerator data placement
target = hcl.Platform.aws_f1
s = hcl.create_schedule([op1, op2], math_func)
s.to(op1, target.xcel)
s.to(op2, target.host)

# test ir correctness 
target.config(compiler="vitis", mode="debug")
code = hcl.build(s, target)
assert "#pragma HLS PIPELINE II=1" in code
os.remove("vadd.cpp")