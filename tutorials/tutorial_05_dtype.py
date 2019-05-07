"""
Data Type Customization
=======================

**Author**: Yi-Hsiang Lai (seanlatias@github)

In this tutorial, we will show the data types supported by HeteroCL. In
addition, we will demonstrate how to apply data type customization in
HeteroCL.
"""

import heterocl as hcl

##############################################################################
# Data Types Supported by HeteroCL
# --------------------------------
# HeteroCL supports both bit-accurate data types and floating points. We show
# some examples below. If no argument is provided, the default bitwidth for
# each type is 32.

hcl.Int(15) # 15-bit signed integer
hcl.UInt(24) # 24-bit unsigned integer
hcl.Fixed(13, 5) # 13-bit signed fixed point with 5 fractional bits
hcl.UFixed(44, 30) # 44-bit unsigned fixed point with 30 fractional bits
hcl.Float(32) # single-precision floating point
hcl.Float(64) # double-precision floating point

##############################################################################
# These data types can be used in ``hcl.init`` to set the default data type

hcl.init(hcl.Float())

##############################################################################
# Data Type Customization
# -----------------------
# Another important hardware customization is data type customization, which
# can be data quantization or downsizing a data type. Data quantization has
# been proved to improve hardware efficiency in many accelerators. In HeteroCL,
# to apply data type customization, we need to use ``hcl.create_scheme``,

A = hcl.placeholder((10,))

def quantization(A):
    return hcl.compute(A.shape, lambda x: hcl.tanh(A[x]), "B")

##############################################################################
# First, let's build the application without applying any quantization scheme.

s = hcl.create_schedule([A], quantization)
f = hcl.build(s)

import numpy as np

hcl_A = hcl.asarray(np.random.rand(10)*2-1)
hcl_B = hcl.asarray(np.zeros(10))

f(hcl_A, hcl_B)

np_A = hcl_A.asnumpy()
np_B = hcl_B.asnumpy()

print('Before tanh')
print(np_A)
print('After tanh')
print(np_B)

##############################################################################
# Now let's use ``hcl.create_scheme`` to create a quantization scheme. The
# usage is the same as ``hcl.create_schedule``.

sm = hcl.create_scheme([A], quantization)
sm_B = quantization.B

##############################################################################
# After we create the schemes, we have two methods that can be used. First,
# if we are dealing with **integers**, we need to use ``downsize``. Second,
# if we are dealing with **floating points**, we need to use ``quantize``.
# No matter which method we choose, the first parameter is a list of tensors
# we want to quantize/downsize and the second parameter is the target data
# type.

sm.quantize(sm_B, hcl.Fixed(10, 8))

##############################################################################
# In this example, since we know the output of `tanh` is between 1 and -1,
# we can safely set the integer part to be 2 bits (i.e., 10-8). The larger
# total bitwidth we choose, the more accurate we get. Now we can create the
# schedule by using ``hcl.create_schedule_from_scheme``, build the executable,
# and test it.

sl = hcl.create_schedule_from_scheme(sm)
f = hcl.build(sl)

hcl_BQ = hcl.asarray(np.zeros(10), dtype = hcl.Fixed(10, 8))

f(hcl_A, hcl_BQ)

np_BQ = hcl_BQ.asnumpy()

print('Without quantization')
print(np_B)
print('Quantized to Fixed(10, 8)')
print(np_BQ)

##############################################################################
# We can double-check this.

assert np.array_equal(np_BQ, np.trunc(np_B*256)/256)
