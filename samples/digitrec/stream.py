import heterocl as hcl
import time
import numpy as np
import math
from digitrec_data import read_digitrec_data

N = 8 * 8
max_bit = int(math.ceil(math.log(N, 2)))
test_size = ()
data_size = (10, 1800)

dtype_image = hcl.UInt(N)
dtype_knnmat = hcl.UInt(max_bit)

setting = {
  "version" : "2019.1",
  "clock"   : "10"
}
tool = hcl.tool.vivado("csim", setting)
target = hcl.platform.zc706

def knn(test_images, train_images):

    def popcount(num):
        out = hcl.scalar(0, "out")
        with hcl.for_(0, train_images.type.bits) as i:
            out.v += num[i]
        return out.v


    @hcl.def_([(10, 3), (10,3)])
    def sort_knn(knn_mat, knn_sorted):
        val = hcl.scalar(0, "val")
        with hcl.for_(0,10) as i:
          with hcl.for_(0,3) as j:
            with hcl.if_( j == 2 ):
              with hcl.if_( knn_mat[i][0] > knn_mat[i][1] ):
                val.v = knn_mat[i][0] 
                knn_mat[i][0] = knn_mat[i][1]
                knn_mat[i][1] = val.v
            with hcl.else_():
              with hcl.if_( knn_mat[i][j] > knn_mat[i][j+1] ):
                val.v = knn_mat[i][j] 
                knn_mat[i][j] = knn_mat[i][j+1]
                knn_mat[i][j+1] = val.v
            knn_sorted[i][j] = knn_mat[i][j]

    def knn_vote(knn_mat, j):
        id0 = hcl.scalar(0, "id0")
        id1 = hcl.scalar(0, "id1")
        id2 = hcl.scalar(0, "id2")
        count = hcl.scalar(0, "count")
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

    @hcl.def_([(10,1800), (10,3)])
    def update_knn(dist, knn_mat):
        with hcl.for_(0,10) as i:
          with hcl.for_(0,1800) as j:
            max_id = hcl.scalar(0, "max_id")
            with hcl.for_(0, 3) as k:
              with hcl.if_(knn_mat[i][k] > knn_mat[i][max_id.v]):
                max_id.v = k
            with hcl.if_(dist[i][j] < knn_mat[i][max_id.v]):
              knn_mat[i][max_id.v] = dist[i][j]

    diff = hcl.compute(train_images.shape,
                       lambda x, y: train_images[x][y] ^ test_image,
                       "diff")
    dist = hcl.compute(diff.shape,
                       lambda x, y: popcount(diff[x][y]),
                       "dist")

    knn_mat = hcl.compute((10, 3), lambda x, y: 50, "knn_mat")
    knn_sorted = hcl.compute((10, 3), lambda x, y: 0, "knn_sorted")
    update_knn(dist, knn_mat)
    sort_knn(knn_mat, knn_sorted)
    knn_pred = hcl.compute((10,), 
                           lambda x: knn_vote(knn_sorted, x), "vote")
    return knn_pred

test_image = hcl.placeholder((), "test_image")
train_images = hcl.placeholder(data_size, "train_images", dtype_image)

scheme = hcl.create_scheme([test_image, train_images], knn)
scheme.downsize([knn.dist, knn.dist.out, knn.knn_mat], dtype_knnmat)

s = hcl.create_schedule_from_scheme(scheme)

diff = knn.diff
dist = knn.dist
# vote = knn.copy
# knn_update = knn.knn_update

# s.to([train_images], target.xcel)
# s.to(vote, target.host)

# merge loop nests
# s[diff].compute_at(s[dist], dist.axis[1])
# s[dist].compute_at(s[knn_update], knn_update.axis[1])

# reorder loop to expose more parallelism
# s[knn_update].reorder(knn_update.axis[1], knn_update.axis[0])

# parallel outer loop and pipeline inner loop
# s[knn_update].parallel(knn_update.axis[1])
# s[knn_update].pipeline(knn_update.axis[0])

# at the end, we build the whole offloaded function.
# print(hcl.lower(s))
f = hcl.build(s)

train_images, _, test_images, test_labels = read_digitrec_data()
total = len(test_images)
total_time = 0

# read returned prediction from streaming pipe
hcl_train_images = hcl.asarray(train_images, dtype_image)
hcl_knn_pred = hcl.asarray(np.zeros((10,)))

start = time.time()
f(test_images[0], hcl_train_images, hcl_knn_pred)
total_time = total_time + (time.time() - start)

knn_result = hcl_knn_pred.asnumpy()

correct = 0.0
# for i in range(total):
#     if np.argmax(knn_result[i]) == test_labels[i]:
#         correct += 1

print("Average kernel time (s): {:.2f}".format(total_time/total))
print("Accuracy (%): {:.2f}".format(100*correct/1))
