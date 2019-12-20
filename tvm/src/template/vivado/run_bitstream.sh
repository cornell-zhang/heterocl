#!/bin/bash

#-------------------------------------------------------------------------
# Variables
#-------------------------------------------------------------------------

# Xillybus files 
XILLYBUS_TAR="xillybus.tgz"
WORKDIR="./zedboard_project"

# HLS generated verilog files directory
VERILOG_DIR="bnn.prj/solution1/syn/verilog"

# The generated bitstream file
GENFILE="$WORKDIR/xillybus/vivado/xillydemo.runs/impl_1/xillydemo.bit"
OUTFILE="xillydemo.bit"

# Check here for synthesis errors
LOGFILE="$WORKDIR/xillybus/vivado/xillydemo.runs/synth_1/runme.log"

#-------------------------------------------------------------------------
# checks
#-------------------------------------------------------------------------

if [ ! -d $VERILOG_DIR ]; then
  printf "Cannot find directory $VERILOG_DIR, which holds the HLS generated Verilog.\n"
  exit
fi

if [ ! -f $XILLYBUS_TAR ]; then
  printf "$XILLYBUS_TAR\n"
  printf "Cannot find FPGA wrapper directory! Make sure you are running on amdpool.\n"
  exit
fi

#-----------------------------------------------------------------------
# Run the xillybus flow
#-----------------------------------------------------------------------

# copy xillybus files if they are not already there
if [ ! -d $WORKDIR ]; then
  printf "Copying zedboard files to current directory.\n"
  cp -r $XILLYBUS_TAR .
  tar -xzf $XILLYBUS_TAR
  rm -f $WORKDIR.tar.gz
  if [ ! -d $WORKDIR ]; then
    printf "Error after extracting zedboard files.\n"
    exit
  fi
fi

# copy generate files to appropriate directory
printf "Copying Verilog files from $VERILOG_DIR.\n"
rm -f $WORKDIR/xillybus/src/fpga-design/*
cp $VERILOG_DIR/* $WORKDIR/xillybus/src/fpga-design

# run vivado
printf "Running vivado\n"
pushd $WORKDIR/xillybus
vivado -mode batch -source xillydemo-vivado.tcl
rm -f *.backup.log *.jou
popd

if [ -e $GENFILE ]; then
  printf "\nBitstream successfuly generated.\n\n"
  printf "Copying $OUTFILE to current directory.\n\n"
  cp $GENFILE $OUTFILE
else
  printf "\n**** There were errors creating the bitstream!\n\n"
  grep --color=always -i error $LOGFILE
fi

