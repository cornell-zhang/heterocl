"""
Custom Module Definition
========================

**Author**: Yi-Hsiang Lai (yl2666@cornell.edu)

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

hcl.init()

def maximum(A, B, C, D):

    @hcl.module([(10,), (10,), ()])
    def find_max(A, B, x):
        with hcl.if_(A[x] > B[x]):
            hcl.return_(A[x])
        with hcl.else_():
            hcl.return_(B[x])

##############################################################################
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

print(hcl_A)
print(hcl_B)
print(hcl_C)
print(hcl_D)
print(hcl_O)
