#===============================================================#
#                                                               #
#                       	run_hw.sh                         	#
#                                                               #
#   	A bash script to synthesize and generate bitstream 			#
#																#
#                                                               #
#===============================================================#


#!/bin/bash
make clean

# the k value of KNN, default is 3
k_value=3
# the directory of this lab
app_dir=`pwd`

### COMPILATION
# create some blank-line space for easy readability
echo ""; echo ""; echo "" ; echo ""
echo "####################################################"
echo " Synthesize and Generate Bitstream with K_CONST=$k_value"
echo "####################################################"
make ocl OCL_TARGET=hw OCL_PLATFORM=$AWS_PLATFORM APPLICATION_DIR=$app_dir K_CONST=$k_value
#export XCL_EMULATION_MODE=hw_emu
#./DigitRec_host.exe -f DigitRec.hw_emu.xclbin 

