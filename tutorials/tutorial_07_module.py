"""
Custom Module Definition
========================

**Author**: Yi-Hsiang Lai (seanlatias@github.com)

In this tutorial, we will introduce a new API called ``module``, which allows
users to define a hardware module.
"""

import heterocl as hcl
import numpy as np

##############################################################################
# Defining a Hardware Module
# --------------------------
# It is important for users to define a hardware module. The main reason is
# that by reusing the defined hardware module, we can reduce the resource
# usage of the design. To define a module, what we need to do is to define a
# Python function. Then, apply the function with a decorator. Within the
# decorator, we need to specify the shapes of the arguments. Following we show
# an example of defining a hardware module that return the maximum value of
# two tensors with a given index.
#
# Note that in this example, we have three input arguments, which are `A`, `B`,
# and `x`. The first two arguments are tensors with shape `(10,)` while the
# last argument is a variable. To represent the shape of a variable, we use an
# empty tuple `()`.
#
# Another thing to be noted is that we use ``hcl.return_`` for the return
# value. We can see that we can have multiple `return` statements.
#
# Use the Defined Module
# ----------------------
# To use the module, it is just like a normal Python call. There is nothing
# special here. Following we show an example of finding the element-wise
# maximum value of four tensors.

hcl.init()

def maximum(A, B, C, D):

    @hcl.def_([A.shape, B.shape, ()])
    def find_max(A, B, x):
        with hcl.if_(A[x] > B[x]):
            hcl.return_(A[x])
        with hcl.else_():
            hcl.return_(B[x])

    max_1 = hcl.compute(A.shape, lambda x: find_max(A, B, x), "max_1")
    max_2 = hcl.compute(A.shape, lambda x: find_max(C, D, x), "max_2")
    return hcl.compute(A.shape, lambda x: find_max(max_1, max_2, x), "max_o")

##############################################################################
# We can first inspect the generated IR. You can see that for each computation,
# we reuse the same module to find the maximum.

A = hcl.placeholder((10,), "A")
B = hcl.placeholder((10,), "B")
C = hcl.placeholder((10,), "C")
D = hcl.placeholder((10,), "D")

s = hcl.create_schedule([A, B, C, D], maximum)
print(hcl.lower(s))

##############################################################################
# Finally, let's run the algorithm and check the results

f = hcl.build(s)

a = np.random.randint(100, size=(10,))
b = np.random.randint(100, size=(10,))
c = np.random.randint(100, size=(10,))
d = np.random.randint(100, size=(10,))
o = np.zeros(10)

hcl_A = hcl.asarray(a)
hcl_B = hcl.asarray(b)
hcl_C = hcl.asarray(c)
hcl_D = hcl.asarray(d)
hcl_O = hcl.asarray(o, dtype=hcl.Int())

f(hcl_A, hcl_B, hcl_C, hcl_D, hcl_O)

print("Input tensors:")
print(hcl_A)
print(hcl_B)
print(hcl_C)
print(hcl_D)
print("Output tensor:")
print(hcl_O)

# Test the correctness
m1 = np.maximum(a, b)
m2 = np.maximum(c, d)
m = np.maximum(m1, m2)
assert np.array_equal(hcl_O.asnumpy(), m)

##############################################################################
# Modules Without Return Statement
# --------------------------------
# HeteroCL also allows users to define modules without a return statement. The
# usage is exactly the same as what we just introduced. The only differece is
# that the module can be called in a stand-alone way. Namely, it does not need
# to be contained in any HeteroCL APIs. Let's use the same example of finding
# the maximum. However, this time we update the output directly.

hcl.init()

def maximum2(A, B, C, D):

    # B will be the tensor that holds the maximum values
    @hcl.def_([A.shape, B.shape])
    def find_max(A, B):
        with hcl.for_(0, A.shape[0]) as i:
            with hcl.if_(A[i] > B[i]):
                B[i] = A[i]

    find_max(A, B)
    find_max(C, D)
    find_max(B, D)

s = hcl.create_schedule([A, B, C, D], maximum2)
f = hcl.build(s)

##############################################################################
# In the above example, we can see that now without the return value, we can
# directly call the defined module. Let's check the results. They should be
# the same as our first example.

f(hcl_A, hcl_B, hcl_C, hcl_D)

print("Output tensor:")
print(hcl_D)

# Test the correctness
m1 = np.maximum(a, b)
m2 = np.maximum(c, d)
m = np.maximum(m1, m2)
assert np.array_equal(hcl_D.asnumpy(), m)

##############################################################################
# Data Type Customization for Modules
# -----------------------------------
# We can also apply data type customization to our defined modules. There are
# two ways to do that. First, you can specify the data types directly in the
# module decorator. Second, you can use the ``quantize`` and ``downsize`` APIs.
# Let's show how we can downsize the first example.

A = hcl.placeholder((10,), dtype=hcl.UInt(4))
B = hcl.placeholder((10,), dtype=hcl.UInt(4))
C = hcl.placeholder((10,), dtype=hcl.UInt(4))
D = hcl.placeholder((10,), dtype=hcl.UInt(4))

s = hcl.create_scheme([A, B, C, D], maximum)
# Downsize the input arguments and also the return value
s.downsize([maximum.find_max.A, maximum.find_max.B, maximum.find_max], hcl.UInt(4))
# We also need to downsize the intermediate results
s.downsize([maximum.max_1, maximum.max_2], hcl.UInt(4))
s = hcl.create_schedule_from_scheme(s)
f = hcl.build(s)

##############################################################################
# Let's run it.

hcl_A = hcl.asarray(a, hcl.UInt(4))
hcl_B = hcl.asarray(b, hcl.UInt(4))
hcl_C = hcl.asarray(c, hcl.UInt(4))
hcl_D = hcl.asarray(d, hcl.UInt(4))
hcl_O = hcl.asarray(o)

f(hcl_A, hcl_B, hcl_C, hcl_D, hcl_O)

print("Downsized output tensor:")
print(hcl_O)

##############################################################################
# We can see that the results are downsized to 4-bit numbers. We can double
# check this.

# Test the correctness
m1 = np.maximum(a%16, b%16)
m2 = np.maximum(c%16, d%16)
m = np.maximum(m1%16, m2%16)
assert np.array_equal(hcl_O.asnumpy(), m)
