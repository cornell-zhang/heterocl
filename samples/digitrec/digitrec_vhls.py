import heterocl as hcl
import time
import numpy as np
import math
from digitrec_data import read_digitrec_data

N = 8 * 8
max_bit = int(math.ceil(math.log(N, 2)))
data_size = (10, 1800)

dtype_image = hcl.UInt(N)
dtype_knnmat = hcl.UInt(max_bit)

def knn(test_image, train_images):

    # Imperative programming and bit operations (§2)
    def popcount(num):
        out = hcl.local(0, "out")
        with hcl.for_(0, train_images.type.bits) as i:
            # Bit selection operation
            out.v += num[i]
        return out.v

    # This function update the candidates, i.e., `knn_mat`. Here we mutate
    # through the shape of tensor `dist`. For each `dist` value, if it is
    # smaller than the maximum candidate, we replace it.
    def update_knn(dist, knn_mat, i, j):
        max_id = hcl.local(0, "max_id")
        with hcl.for_(0, 3) as k:
            with hcl.if_(knn_mat[i][k] > knn_mat[i][max_id.v]):
                max_id.v = k
        with hcl.if_(dist[i][j] < knn_mat[i][max_id.v]):
            knn_mat[i][max_id.v] = dist[i][j]

    # This function sorts the 10 x 3 matrix. Sorting each row elements from 
    # small to large distance, and find digit index where the distance is the smallest
    # then returns the digit having the highest scores
    def sort_knn(knn_mat, i, j):
        val = hcl.local(0, "val")
        with hcl.if_( j == 1 ):
            with hcl.if_( knn_mat[i][1] > knn_mat[i][2] ):
                val.v = knn_mat[i][1] 
                knn_mat[i][1] = knn_mat[i][2]
                knn_mat[i][2] = val.v
        with hcl.else_():
            with hcl.if_( knn_mat[i][0] > knn_mat[i][1] ):
                val.v = knn_mat[i][0] 
                knn_mat[i][0] = knn_mat[i][1]
                knn_mat[i][1] = val.v

    def knn_vote(knn_mat, j):
        id0 = hcl.local(0, "id0")
        id1 = hcl.local(0, "id1")
        id2 = hcl.local(0, "id2")
        count = hcl.local(0, "count")
        with hcl.for_(0, 10) as n:
            with hcl.if_(knn_mat[n][0] < knn_mat[id0.v][0]):
                id0.v = n
        with hcl.for_(0, 10) as m:
            with hcl.if_(knn_mat[m][0] < knn_mat[id1.v][0]):
                id1.v = m
        with hcl.for_(0, 10) as k:
            with hcl.if_(knn_mat[k][0] < knn_mat[id2.v][0]):
                id2.v = k
        with hcl.if_(j == id0.v):
            count.v += 1 
        with hcl.elif_(j == id1.v):
            count.v += 1 
        with hcl.elif_(j == id2.v):
            count.v += 1 
        with hcl.else_():
            count.v += 0 
        return count.v

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

    # Fifth step: voting candidates (§3.5)
    hcl.mutate((10, 3), lambda x, y: sort_knn(knn_mat, x, y), "sort")

    # Sixth step: compute the score baord ranking 
    knn_new = hcl.compute(knn_mat.shape, lambda x, y: knn_mat[x][y], "new")
    knn_pred = hcl.compute((10,), lambda x: knn_vote(knn_mat, x), "vote")

    # computed data 
    return knn_pred

# Inputs/Outputs definition (§4)
# Scalars (§4.1)
test_image = hcl.placeholder((), "test_image", dtype_image)
# Tensors (§4.2)
train_images = hcl.placeholder(data_size, "train_images", dtype_image)

# Data type customization (§5.1)
scheme = hcl.create_scheme([test_image, train_images], knn)
scheme.downsize([knn.dist, knn.dist.out, knn.knn_mat], dtype_knnmat)

# Compute customization (§5.2)
s = hcl.create_schedule_from_scheme(scheme)

diff = knn.diff
dist = knn.dist
vote = knn.new
knn_update = knn.knn_update

# s.stream_to(test_image, hcl.FPGA("intel"))
s.to(train_images, hcl.FPGA("intel"))
s.to(vote, hcl.CPU("x86"))

# Merge loop nests
s[diff].compute_at(s[dist], dist.axis[1])
s[dist].compute_at(s[knn_update], knn_update.axis[1])

# Reorder loop to expose more parallelism
s[knn_update].reorder(knn_update.axis[1], knn_update.axis[0])

# Parallel outer loop and pipeline inner loop
s[knn_update].parallel(knn_update.axis[1])
s[knn_update].pipeline(knn_update.axis[0])

# At the end, we build the whole offloaded function.
print(hcl.lower(s))
target = hcl.env.aws_f1
# target.tool.mode = "sim/impl" 
# hcl.sim / sw
# hcl.impl # refer stage -> tool opt cli
# target.tool[''] 
# target.host["lang" "compiler"]
# targte.host
# target.xcel # 
f = hcl.build(s, target)

# print(f)
# import sys; sys.exit(1)

train_images, _, test_images, test_labels = read_digitrec_data()
correct = 0.0
total_time = 0
for i in range(0, 1):

    hcl_train_images = hcl.asarray(train_images, dtype_image)
    hcl_knn_pred = hcl.asarray(np.zeros((10,)), dtype_knnmat)

    start = time.time()
    f(test_images[i], hcl_train_images, hcl_knn_pred)
    total_time = total_time + (time.time() - start)

    knn_mat = hcl_knn_pred.asnumpy()
    print(knn_mat)

    if knn_mat == test_labels[i]:
        correct += 1

print("Average kernel time (s): {:.2f}".format(total_time/1))
print("Accuracy (%): {:.2f}".format(100*correct/1))
