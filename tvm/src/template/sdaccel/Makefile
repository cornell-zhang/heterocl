# Set kernel name
KERNEL_NAME = top_function_0

# Set host source and headers (xcpp)
HOST_SRC_CPP = host.cpp utils.cpp 
HOST_SRC_H = utils.h

# Set host code include paths
HOST_INC = -I$(XILINX_VIVADO)/include -I/opt/xilinx/Xilinx_SDx_2018.2_0614_1954/Vivado/2018.2/include
HOST_LIB = -L$(XILINX_VIVADO)/lib -L/opt/xilinx/Xilinx_SDx_2018.2_0614_1954/Vivado/2018.2/lib/

# Set kernel file (xocc)
OCL_KERNEL_SRC = kernel.cl
OCL_KERNEL_H = 
APPLICATION_DIR = $(shell pwd)

# Set opencl kernel arguments
OCL_KERNEL_ARGS = --max_memory_ports all --report system

#-------------------------
# Leave the rest to harness
#-------------------------
include harness.mk

