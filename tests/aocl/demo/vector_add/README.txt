#Compiling for Emulator
aoc -march=emulator device/vector_add.cl -I $INTELFPGAOCLSDKROOT/include/kernel_headers -o bin/vector_add.aocx

#Compiling the Host Program
make

#Running with the Emulator 
CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 bin/host 