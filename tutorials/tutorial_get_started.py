"""
Getting Started
===============

**Author**: Yi-Hsiang Lai (yl2666@cornell.edu)

In this tutorial we will demonstrate the basic usage of HeteroCL. HeteroCL is
a programming model that provides an abstraction that captures the hardware
customization of heterogeneous devices.
"""

##############################################################################
# Import HeteroCL
# ---------------
# We usually use `hcl` as the acronym of HeteroCL.

import heterocl as hcl

##############################################################################
# Initialize the Environment
# --------------------------
# We need to initialize the environment for each HeteroCL application. We can
# do this by calling the API `hcl.init()`. We can also set the default data
# type for every computation via this API. The default data type is **32-bit**
# integers.
#
# .. note::
#
#    For more information on the data types, please see
#    :ref:`sphx_glr_tutorials_tutorial_dtype.py`.

hcl.init()

##############################################################################
# Inputs/Outpus Definition
# ------------------------
# After we initialize the algorithm, we need to define the inputs/outpues to
# our application. HeteroCL provides two types of inputs/outputs, which are
# `hcl.var` and `hcl.placeholer`. The former is a **scalar** and can **only be
# used as an input**, while the latter is a **tensor** that can be served as
# both an input and an output. For both APIs, we can set their name and data
# types. Both are optional. If the data type is not specified, the default
# data type set in `hcl.init` will be applied. In addition, since
# `hcl.placeholder`, we need to specify the shape of it. In this example,
# we declare a scalar input `a` and a two-dimensional tensor input `A` with
# shape `(10, 10)`.
#
# .. note::
#
#    For more information on the interfaces, please see
#    :obj:`heterocl.api.var` and :obj:`heterocl.api.placeholder`

a = hcl.var("a")
A = hcl.placeholder((10, 10), "A")

##############################################################################
# Algorithm Definition
# --------------------
# To define an algorithm in HeteroCL, what we need to do is to define a Python
# function, whose arguments are the inputs/outputs. In this example, we define
# a function name `simple_compute` with arguments `a` and `A`. What we define
# in the function here is a computation that adda the value of `a` to each
# element of `A`. Since the computation is very regular, we can describe it
# using vector-code programming. HeteroCL provides several APIs that can
# describe such type of computations. One example is `hcl.compute`. Finally,
# we return the computed tensor as the output of the Python function.

def simple_compute(a, A):

    B = hcl.compute(A.shape, lambda x, y: A[x, y] + a, "B")
    """
    The above API is equivalent to the following Python code.

    for x in range(0, 10):
        for y in range(0, 10):
            B[x, y] = A[x, y] + a
    """

    return B

##############################################################################
# Apply Hardware Customization
# ----------------------------
# Usually, our next step is apply various hardware customization to the
# application. In this tutorial, we skip this step which will be discuss in
# the later tutorials. However, we still need to build a default schedule
# by using `hcl.create_schedule` whose input is a list of inputs/outputs and
# the Python function that defines the algorithm.

s = hcl.create_schedule([a, A], simple_compute)

##############################################################################
# Inspect the Intermediate Representation (IR)
# --------------------------------------------
# A HeteroCL program will be lowered to an IR before backend code generation.
# HeteroCL provides an API for users to inspect the lowered IR. This could be
# helpful for debugging.

print hcl.lower(s)

##############################################################################
# Create the Executable
# ---------------------
# The next step is to build the executable by using `hcl.build`. You can
# define the target of the executable, where the default target is `llvm`.
# Namely, the executable will be run on CPU. The input for this API is the
# schedule we just created.

f = hcl.build(s)

##############################################################################
# Prepare the Inputs/Outputs for the Executable
# ---------------------------------------------
# To run the generated executable, we can feed it with Numpy arrays by using
# `hcl.asarray`. This API will transform a Numpy array to a HeteroCL container
# that can be used as inputs/outputs to the executable. In this tutorial, we
# randomly generate the values for our tensor `A`. Note that in this tutorial,
# since we return a new tensor at the end of our algorithm, we also need to
# prepare tensor `B`.

import numpy as np

hcl_a = 10
hcl_A = hcl.asarray(np.random.randint(100, size = A.shape))
hcl_B = hcl.asarray(np.zeros(A.shape))

##############################################################################
# Run the Executable
# ------------------
# With the prepared inputs/outputs, we can finally feed them to our executable.

f(hcl_a, hcl_A, hcl_B)

##############################################################################
# View the Results
# ----------------
# To view the results, we can transform the HeteroCL tensors back to Numpy
# arrays by using `asnumpy()`.

print hcl_a
print hcl_A.asnumpy()
print hcl_B.asnumpy()

