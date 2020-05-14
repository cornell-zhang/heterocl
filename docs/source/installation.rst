Installation
============

Install with Conda (Recommended)
---------------------------------
First install conda (Anaconda or miniconda) and create a virtual environment with python version >= 3.6.

.. code:: bash
    conda create --name test python=3.6
    conda activate test

After activating the conda environment, you can install the pre-built heterocl library and python packages.

.. code:: bash
    conda install -c cornell-zhang heterocl


Build and Install from Source
-----------------------------
To install HeteroCL, simply clone it from the GitHub. We recommend installing conda first before installing HeteroCL.

.. code:: bash
   
    git clone --recursive https://github.com/cornell-zhang/heterocl.git

After that, go to the downloaded directory and make it.

.. code:: bash

    make -j8

Options
-------

1. You can set your own CMake or LLVM version by setting the PATH in ``Makefile.config``.
2. You can turn off the VHLS C simulation feature by setting ``USE_VIVADO_HLS`` to 0 in ``Makefile.config``.
3. Set ``PYTHONPATH`` environment variable to the HeteroCL python library for development.

.. code:: bash
    export PYTHONPATH=$HCL_HOME/python
