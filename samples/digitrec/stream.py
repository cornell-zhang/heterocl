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
train_data, _, test_images, test_labels = read_digitrec_data()

def knn(test_image, train_images):

    def popcount(num):
        out = hcl.scalar(0, "out")
        with hcl.for_(0, train_images.type.bits) as i:
            out.v += num[i]
        return out.v

    @hcl.def_([(10,3), (10,3)])
    def sort_knn(knn_mat, knn_sorted):
        val = hcl.scalar(0, "val")
        with hcl.for_(0,10, name="i1") as i:
          with hcl.for_(0,3, name="j1") as j:
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
        with hcl.for_(0,10, name="i") as i:
          with hcl.for_(0,1800, name="j") as j:
            max_id = hcl.scalar(0, "max_id")
            with hcl.for_(0, 3, name="k") as k:
              with hcl.if_(knn_mat[i][k] > knn_mat[i][max_id.v]):
                max_id.v = k
            with hcl.if_(dist[i][j] < knn_mat[i][max_id.v]):
              knn_mat[i][max_id.v] = dist[i][j]

    diff = hcl.compute(train_images.shape,
                       lambda x, y: train_images[x][y] ^ test_image, "diff")
    dist = hcl.compute(diff.shape,
                       lambda x, y: popcount(diff[x][y]), "dist")

    knn_mat = hcl.compute((10, 3), lambda x, y: 50, "mat")
    knn_sorted = hcl.compute((10, 3), lambda x, y: 0, "sorted")
    update_knn(dist, knn_mat)
    sort_knn(knn_mat, knn_sorted)
    knn_pred = hcl.compute((10,), 
                   lambda x: knn_vote(knn_sorted, x), "vote")
    return knn_pred

def test_target(target, stream=False):
    test_image = hcl.placeholder((), "test_image")
    train_images = hcl.placeholder(data_size, "train_images", dtype_image)
    scheme = hcl.create_scheme([test_image, train_images], knn)
    if target != "llvm" and False:
        scheme.downsize([knn.dist, knn.dist.out, knn.mat], dtype_knnmat)
    s = hcl.create_schedule_from_scheme(scheme)
    
    diff = knn.diff
    dist = knn.dist
    vote = knn.vote
    update = knn.update_knn
    sort   = knn.sort_knn
    
    # apply data movement 
    s.to(train_images, target.xcel)
    s.to(vote, target.host)
    s.to(knn.mat, s[sort], s[update])
    
    # merge loop nests
    s[diff].compute_at(s[dist], dist.axis[1])
    # s[dist].compute_at(s[update], update.j)
    
    # reorder loop to expose more parallelism
    s[update].reorder(update.i, update.j)
    
    # parallel outer loop and pipeline inner loop
    s[update].pipeline(update.i)
    s[update].parallel(update.j)
    
    print(hcl.lower(s))
    f = hcl.build(s, target)

    hcl_train_images = hcl.asarray(train_data, dtype_image)
    hcl_knn_pred = hcl.asarray(np.zeros((10,)))
    
    f(test_images[0], hcl_train_images, hcl_knn_pred)
    knn_result = hcl_knn_pred.asnumpy()
    assert knn_result.argmax(axis=0) == test_labels[0] 

def test_sdsoc(stream=False):
    setting = {
      "version" : "2019.1",
      "clock"   : "10"
    }
    tool = hcl.tool.sdsoc("csim", setting)
    target = hcl.platform.zc706(tool)
    test_target(target, stream=stream)

test_sdsoc()
