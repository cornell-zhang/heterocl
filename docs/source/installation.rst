Installation
============
To install HeteroCL, simply clone it from the GitHub.

.. code:: bash
   
    git clone --recursive https://github.com/cornell-zhang/heterocl.git

After that, go to the downloaded directory and make it.

.. code:: bash

    make -j8

We highly recommend running HeteroCL with Python 3 since Python 2 will be deprecated in 2020.

Options
-------

1. You can set your own CMake or LLVM version by setting the PATH in ``Makefile.config``.
2. You can turn off the VHLS C simulation feature by setting ``USE_VIVADO_HLS`` to 0 in ``Makefile.config``.
