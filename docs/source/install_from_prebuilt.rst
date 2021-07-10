Insrall from Pre-built Images
=============================
With this method, you can easily deploy and try out HeteroCL. However, to use the latest features and patches,
please install from source.

Install with Conda
------------------
First install conda (Anaconda or miniconda) and create an empty virtual environment. It is recommended to install HeteroCL in a new conda env without any other pre-installed packages to avoid potential conflicts. 

.. code:: bash

    conda create --name hcl-env
    conda activate hcl-env

After activating the conda environment, you can install the pre-built heterocl library and python packages.

.. code:: bash

    conda install -c cornell-zhang heterocl


Install with Docker
-------------------
First make sure docker service is activated on your system by running the hello-world docker image example.

.. code:: bash
   
    docker run hello-world

Then pull back the docker image from DockerHub, and run it using interactive mode. Conda virtual env is pre-installed in the docker image. After activating the conda env, you can use the HeteroCL package.

.. code:: bash

    docker pull hecmay/heterocl:0.3
    docker run -it hecmay/heterocl:0.3 bash

    source /opt/conda/etc/profile.d/conda.sh
    conda activate py36
