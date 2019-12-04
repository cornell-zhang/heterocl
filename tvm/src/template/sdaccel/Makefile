# Set kernel name
KERNEL_NAME = App

# Set host source and headers
# HOST_SRC_CPP = ./src/host/digit_recognition.cpp ./src/host/utils.cpp ./src/host/check_result.cpp
HOST_SRC_CPP = host.cpp utils.cpp 
# HOST_SRC_H   = ./src/host/utils.h ./src/host/check_result.h ./src/host/typedefs.h ./src/host/testing_data.h \
               ./src/host/training_data.h
HOST_SRC_H = utils.h
# DATA         = ./data/*.dat


# Set host code include paths
HOST_INC = -I/opt/Xilinx/Vivado/2018.2.op2258646/include/
HOST_LIB = -L/opt/Xilinx/Vivado/2018.2.op2258646/lib/

# Set kernel file
OCL_KERNEL_SRC = interface.cpp
# OCL_KERNEL_H = ./src/host/typedefs.h
# SDSOC_KERNEL_SRC = ./src/sdsoc/digitrec.cpp
# SDSOC_KERNEL_H = ./src/host/typedefs.h
# SW_KERNEL_SRC = ./src/sw/digitrec_sw.cpp
# SW_KERNEL_H = ./src/host/typedefs.h ./src/sw/digitrec_sw.h

# Set opencl kernel arguments
# log: removed --report system
OCL_KERNEL_ARGS = --max_memory_ports all 

#-------------------------
# Leave the rest to harness
#-------------------------
include harness.mk

