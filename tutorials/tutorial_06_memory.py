"""
Memory Customization
====================

**Author**: Yi-Hsiang Lai (seanlatias@github)

In this tutorial, we demonstrate how memory customization works in HeteroCL.
"""
import heterocl as hcl
import numpy as np
##############################################################################
# Memory Customization in HeteroCL
# --------------------------------
# There are two types of memory customization in HeteroCL. The first one is
# similar to what we have seen in
# :ref:`sphx_glr_tutorials_tutorial_04_compute.py`, where we demonstrate some
# primitives that will be synthesized as pragmas. An example of such primitive
# is ``partition``. Following is an example. Note that the primitive is
# directly applied on the schedule instead of a stage. This is because we are
# modifying the property of a tensor.

hcl.init()

A = hcl.placeholder((10, 10), "A")

def kernel(A):
    return hcl.compute(A.shape, lambda x, y: A[x][y]+1, "B")

s = hcl.create_schedule(A, kernel)
s.partition(A)
print(hcl.lower(s))

##############################################################################
# In the IR, we should see a line that annotates tensor ``A`` to be
# partitioned completely.
#
# .. note::
#
#    For more information, please see
#    :obj:`heterocl.schedule.Schedule.partition`
#
# Data Reuse in HeteroCL
# ======================
# The other type of memory customization primitives involves the introduction
# of allocation of new memory buffers. An example is data reuse. The idea of
# data reuse is to reduce the number of accesses to a tensor by introducing
# an intermediate buffer that holds the values being reused across different
# iterations. This finally leads to better performance in hardware.
#
# Example: 2D Convolution
# -----------------------
# To demonstrate this, we use the computation of 2D convolution as an example.
# Let's see how we can define 2D convolution in HeteroCL.

hcl.init()

A = hcl.placeholder((10, 10), "B")
F = hcl.placeholder((3, 3), "F")

def kernel(A, F):
    r = hcl.reduce_axis(0, 3)
    c = hcl.reduce_axis(0, 3)
    return hcl.compute((8, 8),
            lambda y, x: hcl.sum(A[y+r, x+c]*F[r, c], axis=[r, c]), "B")

s = hcl.create_schedule([A, F], kernel)
print(hcl.lower(s))

##############################################################################
# In the above example, we convolve the input tensor ``A`` with a filter ``F``.
# Then, we store the output in tensor ``B``. Note that the output shape is
# different from the shape of the input tensor. Let's give some real inputs.

hcl_A = hcl.asarray(np.random.randint(0, 10, A.shape))
hcl_F = hcl.asarray(np.random.randint(0, 10, F.shape))
hcl_B = hcl.asarray(np.zeros((8, 8)))
f = hcl.build(s)
f(hcl_A, hcl_F, hcl_B)
print('Input:')
print(hcl_A)
print('Filter:')
print(hcl_F)
print('Output:')
print(hcl_B)

##############################################################################
# To analyze the data reuse, let's take a closer look to the generated IR.
# To begin with, we can see that in two consecutive iterations of ``x`` (i.e.,
# the inner loop), there are 6 pixels that are overlapped, as illustrated in
# the figure below.
#
# .. figure:: moving_x.png
