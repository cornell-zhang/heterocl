"""
Use HeteroCL Platform Features
=======================
**Author**: Hecmay

In this tutorial, we show how to use the pre-defined platforms and configure the corresponding 
toolflows in HeteroCL. The platform class contains a pre-defined list of supported platforms (e.g. 
`hcl.platform.aws_f1` equipped with one VU9P FPGA acceleration card and an Intel Xeon CPU). HeteroCL
allows users to add their own custom platforms with various accelerators. The platform information 
along with data placement scheme will be used to generate target specific backend code.

"""
import numpy as np
import heterocl as hcl

##############################################################################
# Create a program in HeteroCL
# -------------------
# Here we use the stencil kernel from last tutorial to showcase how to use the platform 
# feature to generate host and device code, and run the program in different modes (e.g.
# software simulation, co-simulation or bistream).

def jacobi(input_image):
    def jacobi_kernel(y, x):
        return (input_image[y+1, x-1] +
                input_image[y  , x  ] +
                input_image[y+1, x  ] +
                input_image[y+1, x+1] +
                input_image[y+2, x  ]) / 5

    return hcl.compute(input_image.shape, jacobi_kernel, name="output")

dtype = hcl.Float()
input_image = hcl.placeholder((480, 640), name="input", dtype=dtype)
s = hcl.create_schedule([input_image], jacobi)

##############################################################################
# Use the pre-defined platform 
# -----------------------
# Here we use the AWS F1 platform for demonstration. Users can configure the 
# the toolflow used on the platform. The supported toolflows : "vitis" (Xilinx Vitis), 
# sdsoc(Xilinx SDSoC), "aocl" (Intel AOCL) and "vivado_hls" (Vivado HLS). 
# HeteroCL provides users with four tool modes to facilitate the development process: 
# sw_sim, hw_sim, hw_exe and debug. HeteroCL will return the generated code (i.e. host and device code) 
# in the debug mode, or the compiled function in the other modes. The compiled function is hooked
# with the hardware or software running in the background. Namely, you can invoke FPGA or other 
# accelerators with the python compiled function returned HeteroCL. HeteroCL runtime will handle the 
# the compilation and deployment process based on the platform information.

p = hcl.Platform.aws_f1
p.config(compiler="vitis", mode="debug")

##############################################################################
# Data Movement between Devices 
# --------------------
# The whole HeteroCL program is palced on host by default (as you may notice, 
# HeteroCL returns empty device code in debug mode if no data movement is specified).
# The .to API allows users to move data to different device, and HeteroCL will
# figure out the portiton of the program to be mapped to accelerator. For example, 
# the input image is moved to device scope, and the stencil kernel is executed on
# the device before the output is moved to host.

tensor = jacobi.output
s.to(input_image, p.xcel)
s.to(tensor, p.host)
print(hcl.build(s, p))
