#!/bin/bash
git clone --recursive https://github.com/apache/incubator-tvm tvm-origin
cd tvm-origin
mkdir build
cp ../.circleci/config.cmake build
cd build
~/project/build/pkgs/cmake/build/cmake/bin/cmake ..
make -j4
cd ../python; python setup.py install --user; cd ..
cd topi/python; python setup.py install --user; cd ../../
