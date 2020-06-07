Build and Install from Source
=============================
To build and install HeteroCL from source, please follow the following steps.

1. Create a virtual environment with conda (e.g., Anaconda or miniconda)
2. Download the source code from GitHub. For developers, please **fork** from the repo instead. 

.. code:: bash

    git clone https://github.com/cornell-zhang/heterocl.git

3. (*Optional*) Specify the CMake and/or LLVM path in ``Makefile.config``. For instance

.. code::

    # set your own path to llvm-config
    LLVM_CONFIG = /path/to/llvm-config

4. Run `make` to build and install HeteroCL

.. code:: bash

    make -j8

For Developers
--------------

1. Set the environment variable ``PYTHONPATH`` to reflect the modifications of Python files

.. code:: bash
    export PYTHONPATH=$HCL_HOME/python:$HCL_HOME/hlib/python

2. Run `make` under ``tvm`` to reflect the modification of C source files

.. code:: bash
    make -C tvm

