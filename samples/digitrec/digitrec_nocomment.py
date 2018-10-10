import heterocl as hcl
import tvm
import numpy
from digitrec_data import read_digitrec_data

N = 49
data_size = (10, 1800)
dtype_image = hcl.UInt(N)
dtype_knnmat = hcl.UInt(6)

def top():

  def popcount(num):

    with hcl.CodeBuilder() as cb:
      out = hcl.local(0)
      with cb._for(0, N) as i:
        out[0] += num[i] # Bit selection operation

      hcl.resize(out, hcl.UInt(6))
      return out[0]

  def update_knn(dist, knn_mat, i, j):

    with hcl.CodeBuilder() as cb:
      max_id = hcl.local(0)
      with cb._for(0, 3) as k:
        with cb._if(knn_mat[i][k] > knn_mat[i][max_id]):
          max_id[0] = k
      with cb._if(dist[i][j] < knn_mat[i][max_id]):
        knn_mat[i][max_id] = dist[i][j]

  test_image = hcl.var()
  train_images = hcl.placeholder(data_size)

  diff = hcl.compute(train_images.shape, [train_images], lambda x, y: train_images[x][y] ^ test_image)
  dist = hcl.compute(diff.shape, [diff], lambda x, y: popcount(diff[x][y]))
  knn_mat = hcl.compute((10, 3), [], lambda x, y: 50)
  knn_update = hcl.mut_compute(dist.shape, [dist, knn_mat], lambda x, y: update_knn(dist, knn_mat, x, y))

  hcl.resize([test_image, train_images, diff], dtype_image)
  hcl.resize([dist, knn_mat], dtype_knnmat)

  s = hcl.create_schedule(knn_update)

  return hcl.build(s, [test_image, train_images, knn_mat])

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
for i in range(0, 180):

  hcl_train_images = hcl.asarray(train_images, dtype = dtype_image)
  hcl_knn_mat = hcl.asarray(numpy.zeros((10, 3)), dtype = dtype_knnmat)

  offload(test_images[i], hcl_train_images, hcl_knn_mat)
  knn_mat = hcl_knn_mat.asnumpy()

  if knn_vote(knn_mat) == test_labels[i]:
    correct += 1

print correct/180
