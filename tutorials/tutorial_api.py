"""
HeteroCL Compute APIs
=====================

**Author**: Yi-Hsiang Lai (yl2666@cornell.edu)

In this tutorial, we will show more HeteroCL compute APIs. These APIs are used
to build the algorithm. Note that in HeteroCL, the compute APIs can be used
along with the imperative DSL.
"""

import heterocl as hcl

##############################################################################
# `hcl.compute`
# -------------
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
    return hcl.comptue(A.shape, lambda x: A[x]+B[x], "C")

s = hcl.create_schedule([A, B], compute_example)
print(hcl.lower(s))

##############################################################################
# `hcl.update`
# ------------
# This API is similar to `hcl.compute` in that it defines how you **update a
# tensor** in an elementwise fashion. Note that this API does not return a
# new tensor. More specifically, the return value is `None`.
#
# ``hcl.update(tensor, fupdate, name)``
#
# ``tensor`` is the tesor we want ot update. ``fupate`` is a lambda function
# that describes the elelmentwise update behavior. ``name`` is optional.
