"""
HeteroCL Tutorial : K-Nearest-Neighbor Digit Recognition
========================================================

**Author**: Yi-Hsiang Lai (yl2666@cornell.edu)

HeteroCL is a domain-specific language (DSL) based on TVM that supports
heterogeous backend devices. Moreover, HeteroCL also supports imperative
programming and bit-accurate data types. This tutorials demonstrates how to
implement KNN-based digit recognition in HeteroCL.

Digit recognition is wildly used in many fields, such as automotives.
Despite the fact that most digit recognition algorithms rely on deep learning,
here we simply use the KNN-based algorithm. The input is a 7x7 black and white
image, which are encoded to 1 and 0, respectively. After we have the input test
image, we perform bitwise XOR with all training images. We then calculate the
number of ones for each XOR result. To make the above process easier, we
flatten each 7x7 image to a 49-bit integer, which makes the bitwise XOR faster.
We called the results of number of ones "distance". Our goal is to find the
digit with the smallest distance. In this tutorial, we set K to 3. Namely, for
each digit we will have three candidates. After that, we perform voting
according to the candidates. The winner will be the final label we predict. To
sum up, we can use the following data flow graph to illustrate the whole
process.

.. code-block:: none

        +----------------------+         +--------------------------------------------+
        | 49-bit testing image | xor | 49-bit training images (10 classes x 1800) |
        +----------------------+         +--------------------------------------------+
                                                            |
                                                            V
                         +--------------------------------+
                         | 49-bit diff shape = (10, 1800) |
                         +--------------------------------+
                                                            |  popcount
                                                            V
                             +-----------------------------+    +-------------------------+
                             | distance shape = (10, 1800) |    | knn_mat shape = (10, 3) |
                             +-----------------------------+    +-------------------------+
                                                             |                                                       |
                                                             +-----------------------------+
                                                                                             | update knn_mat
                                                                                             V
                                                            +--------------------------------+
                                                            | updated knn_mat shape = (10,3) |
                                                            +--------------------------------+
                                                                                             | vote
                                                                                             V
                                                                             +--------------+
                                                                             | label (0~10) |
                                                                             +--------------+

In this tutorial, we assume that we want to offload every operation before
voting to FPGA. We create a top function for that accordingly.
"""

# Import necessary modules.
import heterocl as hcl
import time
import numpy as np
import math
from digitrec_data import read_digitrec_data

# Declare some constants and data types. For images, we need unsigned 49-bit
# integers, while for knn matricies, we need unsigned 6-bit integers.
N = 7 * 7
max_bit = int(math.ceil(math.log(N, 2)))
data_size = (10, 1800)

##############################################################################
# Bit Accurate Data Types
# -----------------------
# HeteroCL provides users with a set of bit-accurate data types, which include
# unsigned/signed arbitrary-bit integers and unsigned/signed fixed-points.
# Here we use `UInt(N)` for an N-bit unsigned integer.

dtype_image = hcl.UInt(N)
dtype_knnmat = hcl.UInt(max_bit)

##############################################################################
# Initialize HeteroCL Environment
# -------------------------------
# We can initialize a HeteroCL environment with default data type by using
# `hcl.init(dtype)`. Here we set the default data type of each variable to
# the unsigned integer with the maximum bitwidth.

hcl.init(dtype_image)

# This is the top function we want to offload to FPGA
def top(target=None):

##############################################################################
# Algorithm Definition
# --------------------
# In HeteroCL, we define the algorithm in a separate function call. The
# the arguments are the inputs/outputs. We can also return computed outputs at
# the end of the function call.

    def knn(test_image, train_images):

##############################################################################
# Imperative Programming and Bit Operations
# -----------------------------------------
# This function calculate the number of ones of a 49-bit unsigned integer.
# Here we demonstrate that HeteroCL supports imperative code. All variables
# declared within the block will live in corresponding scope. In this function,
# `out` is an intermediate variable with initial value 0. Since we already set
# the default data type, the data type for `out` is `UInt(N)`. This function
# also shows the capability of bit operations.

        def popcount(num):
            out = hcl.local(0, "out")
            with hcl.for_(0, train_images.type.bits) as i:
                # Bit selection operation
                out[0] += num[i]
            return out[0]

        # This function update the candidates, i.e., `knn_mat`. Here we mutate
        # through the shape of tensor `dist`. For each `dist` value, if it is
        # smaller than the maximum candidate, we replace it.

        def update_knn(dist, knn_mat, i, j):
            max_id = hcl.local(0, "max_id")
            with hcl.for_(0, 3) as k:
                with hcl.if_(knn_mat[i][k] > knn_mat[i][max_id[0]]):
                    max_id[0] = k
            with hcl.if_(dist[i][j] < knn_mat[i][max_id[0]]):
                knn_mat[i][max_id[0]] = dist[i][j]

##############################################################################
# Main Algorithm
# ==============
#
# First Step: XOR
# ---------------
# This is the first step of our algorithm. Namely, compute the XOR of a test
# image with a set of training images. In other words,
#
# `diff[x][y] = train_images[x][y] ^ test_image`,
#
# for all x and y in shape `(10, N)`.
#
# We can use "hcl.compute" to achieve the above computation. This API is
# declarative. Namely, we only specify the results we want, without explicitly
# writing how the results should be computed.
#
# `A = hcl.compute(shape, fcompute, name, dtype)`
#
# The first field is the shape; the second field is a lambda function that
# computes the results for each element of the output tensor. Without applying
# any scheduling function, the code is equivalent to
#
# ..code-block:: python
#     for x in range(0, 10):
#        for y in range(0, 1800):
#            diff[x][y] = train_images[x][y] ^ test_image
#
# It is optional for users to specify the name and output data type. Here we
# do not specify the data type, since by default it is UInt(49).

        diff = hcl.compute(train_images.shape,
                           lambda x, y: train_images[x][y] ^ test_image,
                           "diff")

##############################################################################
# Second Step: Popcount
# ---------------------
# Our next step is to calculate the number of ones for each value in diff.
# This is where we call the function `popcount`. Since what we want to do here
# is similar to the XOR operation above, we can again use `hcl.compute`. Since
# the maximum difference is 49, we only need 6-bit unsigned integers. Here we
# do not specify the data type. We will use "downsize" later.

        dist = hcl.compute(diff.shape,
                           lambda x, y: popcount(diff[x][y]),
                           "dist")

##############################################################################
# Third Step: Initialize the Candidates
# -------------------------------------
# The next step is to compute the candidates. In our algorithm, we find the maximum
        # candidate and replace it if the new incoming value is smaller. Thus, we initialize the
        # value of the candidate tensor with 50, which is larger than the maximum possbile
        # distance: 49. To initialize a tensor we can use still use "hcl.compute" API. Since we
        # do not use any tensor in our compute function, the second field is an empty list.
        knn_mat = hcl.compute((10, 3), lambda x, y: 50, "knn_mat")

        # Fourth step: Update knn_mat
        # ---------------------------------------------------------------------------------------
        # Finally, we update our candidate. Here we can no longer use "hcl.comptue" because we do
        # not update the candidates one by one sequentially. Thus, we use another API called
        # "mut_compute", which compute the lambda function for a given mutation domain. Here we
        # abuse the Python lambda function for simplicity. In Python, a lambda function must
        # return an expression. However, here we return a statement. The code is equivalent to
        # the following Python code.
        #
        # for x in range(0, 10):
        #       for y in range(0, 1800):
        #           update_knn(dist, knn_mat, x, y)
        #
        # The interface is almost the same as "hcl.compute". The only differences are: 1. the
        # shape is the mutation domain instead of the output shape, and 2. since we do not return
        # any new output function, there is no field for the data type.
        #
        # A = hcl.mut_compute(domain, inputs, fcompute, name)
        #
        # The returned value is a Stage, which will be used for scheduling.
        knn_update = hcl.mut_compute(dist.shape, lambda x, y: update_knn(dist, knn_mat, x, y), "knn_update")

        return knn_mat

    # Inputs/Outputs definition
    # ---------------------------------------------------------------------------------------
    # To specify an input variable, we use "hcl.var". We can specify the name and data type
    # of the variable.
    #
    # a = hcl.var(name, dtype)
    #
    # Here the variable is the test image we want to classify. The data type is by default
    # UInt(49)
    test_image = hcl.var("test_image")

    # To specify an input tenosr, we use "hcl.placeholder", which is similar to TVM's API.
    #
    # A = hcl.placeholder(shape, name, dtype)
    #
    # The first field is the shape of the tensor. It is optional for users to set the name
    # and data type. Here the data type is again UInt(49).
    train_images = hcl.placeholder(data_size, "train_images")

    # Specify quantization scheme
    # ---------------------------------------------------------------------------------------
    # This is another feature of HeteroCL, which allows users to specify the data type
    # independently with the algorithm.
    #
    # We can downsize a set of inputs, which can be a placeholder or a variable. Here, we apply
    # the corresponding data type as we mentioned in the previous steps. Note that downsize is
    # used for integers only.
    scheme = hcl.create_scheme([test_image, train_images], knn)
    scheme.downsize([knn.dist, knn.dist.out, knn.knn_mat], dtype_knnmat)

    # Create schedule
    # ---------------------------------------------------------------------------------------
    # All the above describes the algorithm part. Now we can describe how we want to schedule
    # the declarative program.
    s = hcl.create_schedule_from_scheme(scheme)

    # Merge all outer-most loop and parallel it
    # ---------------------------------------------------------------------------------------
    # We can observe that all the operations above iterate through the ten digits, which
    # corresponds to the outer-most loop. We can merge the loops by using compute_at.
    #
    # produce A {
    #       loop_1 {
    #           body_A
    #       }
    # }
    #
    # produce B {
    #       loop_1 {
    #           body_B
    #       }
    # }
    #
    # Since we have a common loop in both stage A and B, we can use compute_at to merge it.
    #
    # s[A].compute_at(s[B], loop_1)
    #
    # This is the equivalent result.
    #
    # produce B {
    #       loop_1 {
    #           produce A {
    #               body_A
    #           }
    #           body_B
    #       }
    # }
    #
    # We can do the same trick on all operations above. Note that we merge all stages to the
    # last stage.

    diff = knn.diff
    dist = knn.dist
    knn_update = knn.knn_update

    s[diff].compute_at(s[knn_update], knn_update.axis[0])
    s[dist].compute_at(s[knn_update], knn_update.axis[0])

    s[knn_update].reorder(knn_update.axis[1], knn_update.axis[0])

    # After we merge the outer-most loop, we can parallel it to make our program faster.
    s[knn_update].parallel(knn_update.axis[1])
    s[knn_update].pipeline(knn_update.axis[0])

    #print hcl.lower(s, [test_image, train_images, knn_mat])

    # At the end, we build the whole offloaded function. It is similar to TVM's interface,
    # where the first field is the schedule and the second field is a list of all inputs and
    # outputs.
    return hcl.build(s, target = target)

# End of top function
###########################################################################################

###########################################################################################
# Main function
# -----------------------------------------------------------------------------------------
# This is the main function. Namely, the complete algorithm we want to run.
# Here we define the data type for images and the candidate tensor

# We get the offloaded function with the provided data types
offload = top()

# Voting algorithm
# -----------------------------------------------------------------------------------------
# This function implements the voting algorithm. We first sort for each digit. After that,
# we compare the values of the first place in each digit. The digit with the shortest value
# get one point. Similarly, we give the point to digits according to their ranking for the
# seoncd place and third place. Finally, we take the digit with the highest point as our
# prediction label.
def knn_vote(knn_mat):
    knn_mat.sort(axis = 1)
    knn_score = np.zeros(10)

    for i in range(0, 3):
        min_id = np.argmin(knn_mat, axis = 0)[i]
        knn_score[min_id] += 1

    return np.argmax(knn_score)

# Data preparation
train_images, _, test_images, test_labels = read_digitrec_data()

# Classification and testing
correct = 0.0

# We have 180 test images
total_time = 0
for i in range(0, 180):

    # Prepare input data to offload function
    # ---------------------------------------------------------------------------------------
    # To load the tensors into the offloaded function, we must first cast it to the correct
    # data type.
    hcl_train_images = hcl.asarray(train_images, dtype_image)
    hcl_knn_mat = hcl.asarray(np.zeros((10, 3)), dtype_knnmat)

    # Execute the offload function and collect the candidates
    start = time.time()
    offload(test_images[i], hcl_train_images, hcl_knn_mat)
    total_time = total_time + (time.time() - start)

    # Convert back to a numpy array
    knn_mat = hcl_knn_mat.asnumpy()

    # Feed the candidates to the voting algorithm and compare the labels
    if knn_vote(knn_mat) == test_labels[i]:
        correct += 1

print "Average kernel time (s): {:.2f}".format(total_time/180)
print "Accuracy (%): {:.2f}".format(100*correct/180)
