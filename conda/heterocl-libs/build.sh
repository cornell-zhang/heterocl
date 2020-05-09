set -e
set -u

# git clone --recursive https://github.com/apache/incubator-tvm tvm-origin
# cd tvm-origin
# git checkout 07ac7712ea2bfecb3c8d21d9358a24c7df585925
# mkdir build
# cp ../.circleci/config.cmake build
# cd build
# cmake ..
# make -j${CPU_COUNT}
# cd ../python; python setup.py install --user; cd ..
# cd topi/python; python setup.py install --user; cd ../../
# 
# cd ../
make -j${CPU_COUNT} VERBOSE=1
