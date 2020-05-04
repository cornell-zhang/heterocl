ifndef XILINX_SDX
$(error Environment variable XILINX_SDX is required and should point to SDAccel install area)
endif
SDA_FLOW = cpu_emu
HOST_SRCS = host.cpp
HOST_EXE_DIR=.
HOST_EXE = host
HOST_CFLAGS = -g -Wall -DFPGA_DEVICE -DC_KERNEL
HOST_LFLAGS = 
KERNEL_SRCS = default_function.cl
KERNEL_NAME = default_function
KERNEL_DEFS =
KERNEL_INCS =
XDEVICE=xilinx:adm-pcie-7v3:1ddr:3.0
XDEVICE_REPO_PATH=
KEEP_TEMP=1
KERNEL_DEBUG=
XCLBIN_NAME=bin_krnl
HOST_CFLAGS+=-DTARGET_DEVICE=\"${XDEVICE}\"
BOARD_SETUP_FILE=setup.sh
ifeq (${SDA_FLOW},cpu_emu)
    CLCC_OPT += -t sw_emu
    XCLBIN = ${XCLBIN_NAME}_cpu_emu.xclbin
else ifeq (${SDA_FLOW},hw_emu)
    CLCC_OPT += -t hw_emu
    XCLBIN = ${XCLBIN_NAME}_hw_emu.xclbin
else ifeq (${SDA_FLOW},hw)
    XCLBIN = ${XCLBIN_NAME}_hw.xclbin
CLCC_OPT += -t hw
endifHOST_ARGS = ${XCLBIN}
COMMON_DIR = ./common
include ${COMMON_DIR}/common.mk
