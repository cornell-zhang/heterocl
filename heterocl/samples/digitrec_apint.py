import heterocl as hcl
import tvm
import numpy
from digitrec_data import read_digitrec_data

from tvm.contrib import util

"""
top function we want to offload
"""
def top():
  def popcount(num, x, y):
    out = 0
    a = num[x][y]
    for i in range(0, 49):
      if a % 2 == 1:
        out = out + 1
      a = a / 2
    return out

  def update_knn(dist, knn_mat):
    for i in range(0, 10):
      for j in range(0, 1800):
        max_id = 0
        for k in range(0, 3):
          max_id = k if knn_mat[i][k] > knn_mat[i][max_id] else max_id
        if dist[i][j] < knn_mat[i][max_id]:
          knn_mat[i][max_id] = dist[i][j]

  test_image = hcl.var(name = "test_image", dtype = "int49")
  train_images = hcl.placeholder((10, 1800), name = "train_images", dtype = "int49")
  diff = hcl.compute(train_images.shape, [test_image, train_images], lambda x, y: train_images[x][y] ^ test_image, name = "diff", dtype = "int49")
  dist = hcl.compute(diff.shape, [diff], lambda x, y: popcount(diff, x, y), name = "dist", extern = [popcount], dtype = "int32")
  knn_mat = hcl.compute((10, 3), [], lambda x, y: 50, name = "knn_mat", dtype = "int32")
  knn_update = hcl.block(update_knn, [dist, knn_mat], name = "knn_update")

  s = tvm.create_schedule(knn_update.op)

  #print tvm.lower(s, [test_image, train_images, knn_mat], simple_mode = True)

  return tvm.build(s, [test_image, train_images, knn_mat])

"""
main function
"""

offload = top()

def knn_vote(knn_mat):
  knn_mat.sort(axis = 1)

  knn_score = numpy.zeros(10)

  for i in range(0, 3):
    min_id = numpy.argmin(knn_mat, axis = 0)[i]
    knn_score[min_id] += 1

  return numpy.argmax(knn_score)

train_images, _, test_images, test_labels = read_digitrec_data()

correct = 0.0

from numbers import Number, Integral

for i in range(0, 180):

  hcl_test_image = test_images[i]
  hcl_train_images = tvm.nd.array(train_images, dtype="int49")
  hcl_knn_mat = tvm.nd.array(numpy.zeros((10, 3)).astype("int32"))

  offload(hcl_test_image, hcl_train_images, hcl_knn_mat)
  knn_mat = hcl_knn_mat.asnumpy()

  if knn_vote(knn_mat) == test_labels[i]:
    correct += 1

print correct/180
