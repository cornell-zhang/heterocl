import heterocl as hcl
import time
import numpy as np
import math
from digitrec_data import read_digitrec_data

# Declare some constants and data types. For images, we need unsigned 49-bit
# integers, while for knn matrices, we need unsigned 6-bit integers.
N = 7 * 7
max_bit = int(math.ceil(math.log(N, 2)))
data_size = (10, 1800)

# HeteroCL provides users with a set of bit-accurate data types, which include
# unsigned/signed arbitrary-bit integers and unsigned/signed fixed-points.
# Here we use `UInt(N)` for an N-bit unsigned integer.
dtype_image = hcl.UInt(N)
dtype_knnmat = hcl.UInt(max_bit)

# We can initialize a HeteroCL environment with default data type by using
# `hcl.init(dtype)`. Here we set the default data type of each variable to
# the unsigned integer with the maximum bitwidth.
hcl.init(dtype_image)


def top(target=None):

    # Algorithm definition (§1)
    def knn(test_image, train_images):

        # Imperative programming and bit operations (§2)
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

        # Main algorithm (§3)
        # Fist step: XOR (§3.1)
        diff = hcl.compute(train_images.shape,
                           lambda x, y: train_images[x][y] ^ test_image,
                           "diff")

        # Second step: popcount (§3.2)
        dist = hcl.compute(diff.shape,
                           lambda x, y: popcount(diff[x][y]),
                           "dist")


        # Third step: initialize the candidates (§3.3)
        knn_mat = hcl.compute((10, 3), lambda x, y: 50, "knn_mat")


        # Fourth step: update the candidates (§3.4)
        hcl.mutate(dist.shape,
                        lambda x, y: update_knn(dist, knn_mat, x, y),
                        "knn_update")

        # Final step: return the candidates (§3.5)
        return knn_mat

    # Inputs/Outputs definition (§4)
    # Scalars (§4.1)
    test_image = hcl.placeholder((), "test_image")
    # Tensors (§4.2)
    train_images = hcl.placeholder(data_size, "train_images")

    # Data type customization (§5.1)
    scheme = hcl.create_scheme([test_image, train_images], knn)
    scheme.downsize([knn.dist, knn.dist.out, knn.knn_mat], dtype_knnmat)

    # Compute customization (§5.2)
    s = hcl.create_schedule_from_scheme(scheme)

    diff = knn.diff
    dist = knn.dist
    knn_update = knn.knn_update

    # Merge loop nests
    s[diff].compute_at(s[dist], dist.axis[1])
    s[dist].compute_at(s[knn_update], knn_update.axis[1])

    # Reorder loop to expose more parallelism
    s[knn_update].reorder(knn_update.axis[1], knn_update.axis[0])

    # Parallel outer loop and pipeline inner loop
    s[knn_update].parallel(knn_update.axis[1])
    s[knn_update].pipeline(knn_update.axis[0])

    # At the end, we build the whole offloaded function.
    return hcl.build(s, target=target)

offload = top('sdaccel')
with open('sdaccel_code.cl', 'w') as f:
  f.write(offload)

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
    # To load the tensors into the offloaded function, we must first cast it to
    # the correct data type.
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

print("Average kernel time (s): {:.2f}".format(total_time/180))
print("Accuracy (%): {:.2f}".format(100*correct/180))

# for testing
assert (correct >= 150.0)
