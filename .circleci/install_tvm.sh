#!/bin/bash
git clone --recursive https://github.com/apache/incubator-tvm tvm-origin
cd tvm-origin
mkdir build
cp ../config.cmake build
cd build
cmake ..
make -j4
