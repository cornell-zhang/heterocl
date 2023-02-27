..  Copyright HeteroCL authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _setup:

############
Installation
############

To install HeteroCL, please make sure you have already cloned the repository and the submodules. This will automatically pull the LLVM repository, which may take a few minutes depending on your network connection. If you want to install LLVM yourself, please refer to the :ref:`developer` section.

.. code-block:: console

  $ git clone https://github.com/cornell-zhang/heterocl.git heterocl-mlir
  $ cd heterocl-mlir
  $ git submodule update --init --recursive

After cloning the submodule, you can install HeteroCL through `pip`. It will install LLVM 15.0.0 and build all the dependencies. Make sure you have `cmake (>=3.19)` and `python (>=3.7)`.

.. code-block:: console

  $ pip install .

Later, you can add the following line to your ``.bashrc`` file to make sure the environment variables are set correctly.

.. code-block:: console

  $ export LLVM_BUILD_DIR=$(pwd)/hcl-dialect/externals/llvm-project/build
  $ export PATH=${LLVM_BUILD_DIR}/bin:${PATH}

To verify HeteroCL is installed correctly, you can run the following test.

.. code-block:: console

  $ python3 -m pytest test


.. _developer:

HeteroCL Developers
-------------------

You can clone and install LLVM in a separate folder, but make sure it is the same version as the one in the submodule (v15.0.0).

.. code-block:: console

  $ git clone https://github.com/llvm/llvm-project.git
  $ cd llvm-project
  $ git checkout tags/llvmorg-15.0.0

Then, you can build LLVM with Python binding using the following commands. It is recommended to install the Python binding in a clean virtual environment.

.. code-block:: console

  $ # Install required packages. Suppose you are inside the llvm-project folder.
  $ python3 -m pip install -r mlir/python/requirements.txt

  $ # Run cmake
  $ mkdir build && cd build
  $ cmake -G "Unix Makefiles" ../llvm \
       -DLLVM_ENABLE_PROJECTS=mlir \
       -DLLVM_BUILD_EXAMPLES=ON \
       -DLLVM_TARGETS_TO_BUILD="host" \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_ENABLE_ASSERTIONS=ON \
       -DLLVM_INSTALL_UTILS=ON \
       -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
       -DPython3_EXECUTABLE=`which python3`
  $ make -j8

  $ # Export the LLVM build directory
  $ export LLVM_BUILD_DIR=$(pwd)

  $ # To enable better backtracing for debugging,
  $ # we suggest setting the following system path
  $ export LLVM_SYMBOLIZER_PATH=$(pwd)/bin/llvm-symbolizer

Clone the HCL-MLIR dialect repository and build it.

.. code-block:: console

  $ git clone git@github.com:cornell-zhang/hcl-dialect.git
  $ cd hcl-dialect
  $ mkdir build && cd build
  $ cmake -G "Unix Makefiles" .. \
       -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
       -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit \
       -DPYTHON_BINDING=ON \
       -DOPENSCOP=OFF \
       -DPython3_EXECUTABLE=`which python3`
  $ make -j8

  $ # Export the generated HCL-MLIR Python library
  $ export PYTHONPATH=$(pwd)/tools/hcl/python_packages/hcl_core:${PYTHONPATH}

Finally, inside the HeteroCL root folder, export the required environment variables.

.. code-block:: console

  $ export PYTHONPATH=$(pwd):${PYTHONPATH}
  $ export PATH=${LLVM_BUILD_DIR}/bin:${PATH}

You can run the following test to make sure everything is set up correctly.

.. code-block:: console

  $ python3 -m pytest test
