import hcl
import tvm
import numpy
from digitrec_data import read_digitrec_data

"""
top function we want to offload
"""
def top(dtype):
  def popcount(num): # Calculate the number of bits
    out = 0
    for i in range(0, 49):
      out = out + num[i] # Bit selection operation
    return out

  def update_knn(dist, knn_mat, i, j): # Mutate update knn_mat
    max_id = 0
    for k in range(0, 3): # find the maximum distance in knn_mat
      max_id = k if knn_mat[i][k] > knn_mat[i][max_id] else max_id
    if dist[i][j] < knn_mat[i][max_id]: # replace the maximum distance if the new one from dist is smaller
      knn_mat[i][max_id] = dist[i][j]

  test_image = hcl.var(name = "test_image", dtype = dtype) # input test image, with bit-accurate data type
  train_images = hcl.placeholder((10, 1800), name = "train_images", dtype = dtype) # input training images, with bit-accurate data type
  # the difference between an input image and a train image, with the same data type as inputs
  diff = hcl.compute(train_images.shape, [test_image, train_images], lambda x, y: train_images[x][y] ^ test_image, name = "diff", dtype = dtype)
  # calculate the distance using popcount
  dist = hcl.compute(diff.shape, [diff], lambda x, y: popcount(diff[x][y]), name = "dist", extern = [popcount], dtype = "int32")
  # initialize a tensor that collects the 3-smallest distances
  knn_mat = hcl.compute((10, 3), [], lambda x, y: 50, name = "knn_mat", dtype = "int32")
  # update knn_mat according to the calculated distance
  knn_update = hcl.mut_compute(dist.shape, [dist, knn_mat], lambda x, y: update_knn(dist, knn_mat, x, y), extern = [update_knn], name = "knn_update")

  s = tvm.create_schedule(knn_update.op)

  return tvm.build(s, [test_image, train_images, knn_mat]) # the offloaded function

"""
main function
"""

dtype = "int49" # the data type for testing images and training images

offload = top(dtype) # get the offload function

def knn_vote(knn_mat): # compute the classified result according to knn_mat
  knn_mat.sort(axis = 1)
  knn_score = numpy.zeros(10)

  for i in range(0, 3):
    min_id = numpy.argmin(knn_mat, axis = 0)[i]
    knn_score[min_id] += 1

  return numpy.argmax(knn_score)

train_images, _, test_images, test_labels = read_digitrec_data() # get testing and training data

correct = 0.0

for i in range(0, 180):

  # convert numpy arrays to tvm arrays
  hcl_train_images = tvm.nd.array(train_images, dtype = dtype)
  hcl_knn_mat = tvm.nd.array(numpy.zeros((10, 3)).astype("int32"))

  # execute the offload function and collect results
  offload(test_images[i], hcl_train_images, hcl_knn_mat)
  knn_mat = hcl_knn_mat.asnumpy() # convert tvm arrays back to numpy arrays

  # compare the results with given labels
  if knn_vote(knn_mat) == test_labels[i]:
    correct += 1

print correct/180
