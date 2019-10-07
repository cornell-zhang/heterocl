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
            key = hcl.scalar(A[i], "key")
            j = hcl.scalar(i-1, "j")
            # while(j >= 0 && key < A[j])
            with hcl.while_(hcl.and_(j >= 0, key < A[j])):
                A[j+1] = A[j]
                j.v -= 1
            A[j+1] = key.v

##############################################################################
# Imperative DSL
# --------------
# To write imperative code in HeteroCL, we need to use a subset of HeteroCL
# DSL, which is *imperative DSL*. HeteroCL's imperative DSL supports a subset
# of Python's control flow statements, including conditional statements and
# control flows. In the above code, we show how we can use ``hcl.for_`` to
# write a `for` loop and ``hcl.while_`` to write a `while` loop. Moreover, we
# use ``hcl.and_`` for logical expressions. Here we also introduce a new API,
# which is ``hcl.scalar``. It is equivalent to
#
# ``hcl.compute((1,))``
#
# Namely, it declares a tensor with exactly one element, which can be treated
# as a **stateful scalar**. Following we show the execution results of the
# implemented sorting algorithm.
#
# .. note::
#
#    Currently we support the following imperative DSLs. Logic operations:
#    :obj:`heterocl.and_`, :obj:`heterocl.or_`. Control flow statements:
#    :obj:`heterocl.if_`, :obj:`heterocl.else_`, :obj:`heterocl.elif_`,
#    :obj:`heterocl.for_`, :obj:`heterocl.while_`, :obj:`heterocl.break_`.

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

##############################################################################
# Bit Operations
# --------------
# HeteroCL also support bit operations including setting/getting a bit/slice
# from a number. This is useful for integer and fixed-point operations.
# Following we show some basic examples.
hcl.init()
A = hcl.placeholder((10,), "A")
def kernel(A):
    # get the LSB of A
    B = hcl.compute(A.shape, lambda x: A[x][0], "B")
    # get the lower 4-bit of A
    C = hcl.compute(A.shape, lambda x: A[x][4:0], "C")
    return B, C

##############################################################################
# Note that for the slicing operations, we follow the convention of Python,
# which is **left exclusive and right inclusive**. Now we can test the results.
s = hcl.create_schedule(A, kernel)
f = hcl.build(s)

np_A = np.random.randint(0, 100, A.shape)
hcl_A = hcl.asarray(np_A)
hcl_B = hcl.asarray(np.zeros(A.shape))
hcl_C = hcl.asarray(np.zeros(A.shape))

f(hcl_A, hcl_B, hcl_C)

print("Input array:")
print(hcl_A)
print("Least-significant bit:")
print(hcl_B)
print("Lower four bits:")
print(hcl_C)

# a simple test
np_B = hcl_B.asnumpy()
np_C = hcl_C.asnumpy()
for i in range(0, 10):
    assert np_B[i] == np_A[i] % 2
    assert np_C[i] == np_A[i] % 16

##############################################################################
# The operations for bit/slice setting is similar. The only difference is that
# we need to use imperative DSL. Following is an example.
hcl.init()
A = hcl.placeholder((10,), "A")
B = hcl.placeholder((10,), "B")
C = hcl.placeholder((10,), "C")
def kernel(A, B, C):
    with hcl.Stage("S"):
        with hcl.for_(0, 10) as i:
            # set the LSB of B to be the same as A
            B[i][0] = A[i][0]
            # set the lower 4-bit of C
            C[i][4:0] = A[i]

s = hcl.create_schedule([A, B, C], kernel)
f = hcl.build(s)
# note that we intentionally limit the range of A
np_A = np.random.randint(0, 16, A.shape)
np_B = np.random.randint(0, 100, A.shape)
np_C = np.random.randint(0, 100, A.shape)
hcl_A = hcl.asarray(np_A)
hcl_B = hcl.asarray(np_B)
hcl_C = hcl.asarray(np_C)

f(hcl_A, hcl_B, hcl_C)

print("Input array:")
print(hcl_A)
print("Before setting the least-significant bit:")
print(np_B)
print("After:")
print(hcl_B)
print("Before setting the lower four bits:")
print(np_C)
print("After:")
print(hcl_C)

# let's do some checks
np_B2 = hcl_B.asnumpy()
np_C2 = hcl_C.asnumpy()

assert np.array_equal(np_B2 % 2, np_A % 2)
assert np.array_equal(np_B2 // 2, np_B // 2)
assert np.array_equal(np_C2 % 16, np_A)
assert np.array_equal(np_C2 // 16, np_C // 16)

