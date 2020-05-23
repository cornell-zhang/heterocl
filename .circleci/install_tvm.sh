#!/bin/bash
git clone --recursive https://github.com/apache/incubator-tvm tvm-origin
cd tvm-origin
git checkout 07ac7712ea2bfecb3c8d21d9358a24c7df585925
# checkout submodules
cd 3rdparty/dmlc-core
git checkout 77df189fdeb1d4628d3218d1d9322fcd84496067
cd ../../
cd 3rdparty/dlpack
git checkout 3ec04430e89a6834e5a1b99471f415fa939bf642
cd ../../
cd 3rdparty/vta-hw
git checkout 21937a067fe0e831244766b41ae915c833ff15ba
cd ../../
cd 3rdparty/rang
git checkout cabe04d6d6b05356fa8f9741704924788f0dd762
cd ../../

mkdir build
cp ../.circleci/config.cmake build
cd build
~/project/build/pkgs/cmake/build/cmake/bin/cmake ..
make -j4
cd ../python; python setup.py install --user; cd ..
cd topi/python; python setup.py install --user; cd ../../
