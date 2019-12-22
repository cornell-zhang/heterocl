SHELL = /bin/bash
VPATH = ./
CC = xcpp
CLCC = xocc
ifeq ($(XDEVICE_REPO_PATH),)
    DEVICE_REPO_OPT = 
else
DEVICE_REPO_OPT = --xp prop:solution.device_repo_paths=${XDEVICE_REPO_PATH}
endif
HOST_CFLAGS += -I${XILINX_SDX}/runtime/include/1_2
HOST_LFLAGS += -L${XILINX_SDX}/runtime/lib/x86_64 -lxilinxopencl -lrt -pthread
CLCC_OPT += $(CLCC_OPT_LEVEL) ${DEVICE_REPO_OPT} --xdevice ${XDEVICE} -o ${XCLBIN} ${KERNEL_DEFS} ${KERNEL_INCS}
ifeq (${KEEP_TEMP},1)
    CLCC_OPT += -s
endif
ifeq (${KERNEL_DEBUG},1)
    CLCC_OPT += -g
endif
CLCC_OPT += --kernel ${KERNEL_NAME}
OBJECTS := $(HOST_SRCS:.cpp=.o)
.PHONY: all
all: run
host: ${HOST_EXE_DIR}/${HOST_EXE}
xbin_cpu_em:
    make SDA_FLOW=cpu_emu xbin -f sdaccel.mk
xbin_hw_em:
    make SDA_FLOW=hw_emu xbin -f sdaccel.mk
xbin_hw :
    make SDA_FLOW=hw xbin -f sdaccel.mk
xbin: ${XCLBIN}
run_cpu_em: 
    make SDA_FLOW=cpu_emu run_em -f sdaccel.mk
run_hw_em: 
    make SDA_FLOW=hw_emu run_em -f sdaccel.mk
run_hw : 
    make SDA_FLOW=hw run_hw_int -f sdaccel.mk
run_em: xconfig host xbin
    XCL_EMULATION_MODE=true ${HOST_EXE_DIR}/${HOST_EXE} ${HOST_ARGS}
run_hw_int : host xbin_hw
    source ${BOARD_SETUP_FILE};${HOST_EXE_DIR}/${HOST_EXE} ${HOST_ARGS}
estimate : 
    ${CLCC} -c -t hw_emu --xdevice ${XDEVICE} --report estimate ${KERNEL_SRCS}
xconfig : emconfig.json
emconfig.json :
    emconfigutil --xdevice ${XDEVICE} ${DEVICE_REPO_OPT} --od .
${HOST_EXE_DIR}/${HOST_EXE} : ${OBJECTS}
    ${CC} ${HOST_LFLAGS} ${OBJECTS} -o $@
${XCLBIN}:
    ${CLCC} ${CLCC_OPT} ${KERNEL_SRCS}
%.o: %.cpp
    ${CC} ${HOST_CFLAGS} -c $< -o $@
clean:
    ${RM} -rf ${HOST_EXE} ${OBJECTS} ${XCLBIN} emconfig.json _xocc_${XCLBIN_NAME}_*.dir .Xil
cleanall: clean
    ${RM} -rf *.xclbin sdaccel_profile_summary.* _xocc_* TempConfig *.log *.jou
