import os
import heterocl as hcl
import numpy as np
from digitrec_data import read_digitrec_data

# Declare some constants and data types. For images, we need unsigned 49-bit integers,
# while for knn matricies, we need unsigned 6-bit integers.
# NOTE: MerlinC currently doesn't support this feature so it will round up to
# a primitve type
N = 7 * 7
data_size = (10, 1800)
dtype_image = hcl.UInt(N)
dtype_knnmat = hcl.UInt(6)

# Set the default data type with maximum precision
hcl.config.init_dtype = dtype_image

# This is the top function we wan to offload to FPGA
def top():

  def popcount(num):
    out = hcl.local(0, "out")
    with hcl.for_(0, N) as i:
      out[0] += num[i] # Bit selection operation
    return out[0]

  def update_knn(dist, knn_mat, i, j):
    max_id = hcl.local(0, "max_id")
    with hcl.for_(0, 3) as k:
      with hcl.if_(knn_mat[i][k] > knn_mat[i][max_id[0]]):
        max_id[0] = k
    with hcl.if_(dist[i][j] < knn_mat[i][max_id[0]]):
      knn_mat[i][max_id[0]] = dist[i][j]

  test_image = hcl.var("test_image")
  train_images = hcl.placeholder(data_size, "train_images")
  diff = hcl.compute(train_images.shape, lambda x, y: train_images[x][y] ^ test_image, "diff")
  dist = hcl.compute(diff.shape, lambda x, y: popcount(diff[x][y]), "dist")
  knn_mat = hcl.compute((10, 3), lambda x, y: 50, "knn_mat")
  knn_update = hcl.mut_compute(dist.shape, lambda x, y: update_knn(dist, knn_mat, x, y), "knn_update")

  hcl.downsize([dist, dist.out, knn_mat], dtype_knnmat)

  s = hcl.create_schedule(knn_update)

  s[diff].compute_at(s[knn_update], knn_update.axis[0])
  s[dist].compute_at(s[knn_update], knn_update.axis[0])
  s[knn_mat].compute_at(s[knn_update], knn_update.axis[0])
  s[knn_update].parallel(knn_update.axis[0])

  return hcl.build(s, [test_image, train_images, knn_mat], target='merlinc')

code = top()
with open('kernel.cpp', 'w') as f:
    f.write(code)
print 'Kernel code is written to kernel.cpp'

# Here we use gcc to evaluate the functionality
os.system('g++ -std=c++11 digitrec_host.cpp kernel.cpp')
os.system('./a.out')
os.system('rm a.out')
