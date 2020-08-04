#!/bin/bash
cd ../tvm/

CXX=/opt/rh/devtoolset-7/root/usr/bin/g++ make -j64
cd -

python test_stream_rewrite.py
