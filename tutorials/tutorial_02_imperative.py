"""
Imperative Programming
======================

**Author**: Yi-Hsiang Lai (seanlatias@github)

There exist many applications that cannot be described using only vectorized
code such as `hcl.compute`. Thus, we introduce imperative programming in
HeteroCL, which makes HeteroCL applications more expressive. In this tutorial,
we will implement *insertion sort* in HeteroCL.
"""

import heterocl as hcl

hcl.init()

A = hcl.placeholder((10,), "A")

##############################################################################
# Stages in HeteroCL
# ------------------
# In HeteroCL, when users write an application, they are actually building a
# compute graph. Each node in a graph is a *stage*. Each edge is directed,
# which represents the data flow between two stages. Some HeteroCL APIs
# naturally form a stage, such as ``hcl.compute``. Since the imperative code
# we are going to write cannot be described using a HeteroCL API, we need to
# wrap it as a stage explicitly via ``hcl.Stage``. Users can specify the name
# of a stage, which is optional. Note that **a HeteroCL application must have
# at least one stage**.

def insertion_sort(A):

    # Introduce a stage.
    with hcl.Stage("S"):
        # for i in range(1, A.shape[0])
        # We can name the axis
        with hcl.for_(1, A.shape[0], name="i") as i:
            key = hcl.local(A[i], "key")
            j = hcl.local(i-1, "j")
            # while(j >= 0 && key < A[j])
            with hcl.while_(hcl.and_(j >= 0, key < A[j])):
                A[j+1] = A[j]
                j[0] -= 1
            A[j+1] = key[0]

##############################################################################
# Imperative DSL
# --------------
# To write imperative code in HeteroCL, we need to use a subset of HeteroCL
# DSL, which is *imperative DSL*. HeteroCL's imperative DSL supports a subset
# of Python's control flow statements, including conditional statements and
# control flows. In the above code, we show how we can use ``hcl.for_`` to
# write a `for` loop and ``hcl.while_`` to write a `while` loop. Moreover, we
# use ``hcl.and_`` for logical expressions. Here we also introduce a new API,
# which is ``hcl.local``. It is equivalent to
#
# ``hcl.compute((1,))``
#
# Namely, it declares a tensor with exactly one element, which can be treated
# as a **stateful scalar**. For a full list of supported semantics, please
# check :ref:`heterocl-dsl`. Following we show the execution results of the
# implemented sorting algorithm.

s = hcl.create_schedule([A], insertion_sort)

##############################################################################
# We can inspect the generated IR.
print(hcl.lower(s))

##############################################################################
# Finally, we build the executable and feed it with Numpy arrays.
f = hcl.build(s)

import numpy as np

hcl_A = hcl.asarray(np.random.randint(50, size=(10,)))

print('Before sorting:')
print(hcl_A)

f(hcl_A)

print('After sorting:')
np_A = hcl_A.asnumpy()
print(np_A)

##############################################################################
# Let's run some tests for verification.
for i in range(1, 10):
    assert np_A[i] >= np_A[i-1]

