"""
HeteroCL Compute APIs
=====================

**Author**: Yi-Hsiang Lai (seanlatias@github)

In this tutorial, we will show more HeteroCL compute APIs. These APIs are used
to build the algorithm. Note that in HeteroCL, the compute APIs can be used
along with the imperative DSL.
"""

import heterocl as hcl

##############################################################################
# ``hcl.compute``
# ---------------
# We have introduced this API before. This API returns a **new tensor** whose
# values are defined in an elementwise fashion. Following we show the API's
# prototype.
#
# ``compute(shape, fcompute, name, dtype)``
#
# ``shape`` defines the shape of the output tensor. ``fcompute`` is a lambda
# function that describes the elementwise definition. ``name`` and ``dtype``
# are optional. We show an example below.

hcl.init()

A = hcl.placeholder((10,), "A")
B = hcl.placeholder((10,), "B")

def compute_example(A, B):
    return hcl.compute(A.shape, lambda x: A[x]+B[x], "C")

s = hcl.create_schedule([A, B], compute_example)
print(hcl.lower(s))

##############################################################################
# ``hcl.update``
# --------------
# This API is similar to `hcl.compute` in that it defines how you **update a
# tensor** in an elementwise fashion. Note that this API does not return a
# new tensor. More specifically, the return value is `None`.
#
# ``hcl.update(tensor, fupdate, name)``
#
# ``tensor`` is the tensor we want ot update. ``fupate`` is a lambda function
# that describes the elelmentwise update behavior. ``name`` is optional. We
# show an example below that does the similar computation as `compute_example`.
# The difference is that instead of returning a new tensor `C`, we send it in
# as an input and update it in place. We can see that the generated IR is
# almost the same.

hcl.init()
A = hcl.placeholder((10,), "A")
B = hcl.placeholder((10,), "B")
C = hcl.placeholder((10,), "C")

def update_example(A, B, C):
    hcl.update(C, lambda x: A[x]+B[x], "U")

s = hcl.create_schedule([A, B, C], update_example)
print(hcl.lower(s))

##############################################################################
# ``hcl.mutate``
# -------------------
# This API allows users to describe any loops with vector code, even if the
# loop body does not have any common pattern or contains imperative DSL.
# This API is useful when we want to perform optimization.
#
# ``hcl.mutate(domain, fbody, name)``
#
# ``domain`` describes the iteration domain of our original `for` loop.
# ``fbody`` is the body statement of the `for` loop. ``name`` is optional. We
# can describe the same computation in the previous two examples using this
# API.

hcl.init()
A = hcl.placeholder((10,), "A")
B = hcl.placeholder((10,), "B")
C = hcl.placeholder((10,), "C")

def mut_example(A, B, C):
    def loop_body(x):
        C[x] = A[x] + B[x]
    hcl.mutate((10,), lambda x: loop_body(x), "M")

s = hcl.create_schedule([A, B, C], mut_example)
print(hcl.lower(s))

##############################################################################
# Note that in this example, we are not allowed to directly write the
# assignment statement inside the lambda function. This is forbidden by Python
# syntax rules.
#
# Combine Imperative DSL with Compute APIs
# ----------------------------------------
# HeteroCL allows users to write a mixed-paradigm programming application.
# This is common when performing reduction operations. Although HeteroCL
# provides APIs for simple reduction operations such as summation and finding
# the maximum number, for more complexed reduction operations such as sorting,
# we need to describe them manually. Following we show an example of finding
# the maximum two values in a tensor.

hcl.init()
A = hcl.placeholder((10,), "A")
M = hcl.placeholder((2,), "M")

def find_max_two(A, M):
    def loop_body(x):
        with hcl.if_(A[x] > M[0]):
            with hcl.if_(A[x] > M[1]):
                M[0] = M[1]
                M[1] = A[x]
            with hcl.else_():
                M[0] = A[x]
    hcl.mutate(A.shape, lambda x: loop_body(x))

s = hcl.create_schedule([A, M], find_max_two)
f = hcl.build(s)

import numpy as np

hcl_A = hcl.asarray(np.random.randint(50, size=(10,)))
hcl_M = hcl.asarray(np.array([-1, -1]))

f(hcl_A, hcl_M)

np_A = hcl_A.asnumpy()
np_M = hcl_M.asnumpy()

print(np_A)
print(np_M)

assert np.array_equal(np_M, np.sort(np_A)[-2:])
