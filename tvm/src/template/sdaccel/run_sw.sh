#===============================================================#
#                                                               #
#                       	run1.sh                         	#
#                                                               #
#   	A bash script to run the software emulation 			#
#																#
#                                                               #
#===============================================================#


#!/bin/bash
make clean

# check env variable setup
if [ -z "$AWS_PLATFORM" ]; then
    echo "AWS_PLATFORM not set up; use default"
    export AWS_PLATFORM=xilinx:adm-pcie-7v3:1ddr:3.0
fi

# set up emulation configuration
echo "#################################################"
echo " Setting emulation configuration..."
echo "#################################################"
export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export XCL_EMULATION_MODE=true
emconfigutil --platform=$AWS_PLATFORM

# the k value of KNN, default is 3
k_value=3
# the directory of this lab
app_dir=`pwd`

### COMPILATION
# create some blank-line space for easy readability
echo ""; echo ""; echo "" ; echo ""
echo "####################################################"
echo " Compiling project with K_CONST=$k_value"
echo "####################################################"
make ocl OCL_TARGET=sw_emu OCL_PLATFORM=$AWS_PLATFORM APPLICATION_DIR=$app_dir K_CONST=$k_value


### EXECUTION
echo ""; echo ""; echo "" ; echo ""
echo "####################################################"
echo " Executing DigitRec with K_CONST=$k_value"
echo "####################################################"
export XCL_EMULATION_MODE=sw_emu
#export XCL_EMULATION_MODE=hw_emu
#./top_function_0_host.exe -f top_function_0.sw_emu.xclbin 

