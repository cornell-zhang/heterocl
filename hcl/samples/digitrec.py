import hcl
import tvm
import numpy
from digitrec_data import read_digitrec_data

def popcount(num, x, y):
  out = 0
  a = num[x][y]
  for i in range(0, 49):
    if a % 2 == 1:
      out = out + 1
    a = a / 2
  return out

def popcount_inline(num, nbits):
  out = 0
  for i in range(0, nbits):
    out = hcl.select(x%2, out+1, out)
    out = out >> 1
  return out

def update_knn(dist, knn_mat):
  for i in range(0, 10):
    for j in range(0, 100):
      max_id = 0
      for k in range(0, 3):
        max_id = k if knn_mat[i][k] > knn_mat[i][max_id] else max_id
      if dist[i][j] < knn_mat[i][max_id]:
        knn_mat[i][max_id] = dist[i][j]

""" function definitons and updates """
input_image = hcl.var(name = "input_image", dtype = "int64")
labelval = hcl.placeholder((10, 1800), name = "labelval", dtype = "int64")
diff = hcl.compute(labelval.shape, [input_image, labelval], lambda x, y: labelval[x][y] ^ input_image, name = "diff", dtype = "int32")
dist = hcl.compute((10, 1800), [diff], lambda x, y: popcount(diff, x, y), name = "dist", inline = False, extern_funcs = [popcount], dtype = "int32")
# alternatively, you can do this
# dist = hcl.compute((10, 100), lambda x, y: popcount_inline(diff[x][y], 49), name = "dist")
knn_mat = hcl.compute((10, 3), [], lambda x, y: 50, name = "knn_mat", dtype = "int32")
knn_update = hcl.block(update_knn, [dist, knn_mat], name = "knn_update")

""" scheduling """
s = tvm.create_schedule(knn_update.op)
"""
# note that diff and dist are Tensor while knn_int and knn_update are Stage
# the axes of a Tensor are the axes of the Tensor's definition
# a.axis == a.stages[0].axis
s[diff].unroll(diff.axis[0]) # diff.axis = [x, y]
s[dist].pipeline(dist.axis[1], II = 1) # dist.axis = [x, y, i]
s[knn_init].unroll(knn_init.axis[0]) # knn_init.axis = [x, y]
s[knn_update].pipeline(knn_update.axis[1], II = 1) # knn_update.axis = [i, j, k]
"""

print tvm.lower(s, [input_image, labelval, knn_mat], simple_mode = True)

train_images, _, test_image, test_label = read_digitrec_data()

""" build the module """
m = tvm.build(s, [input_image, labelval, diff, dist, knn_mat])

""" test the module """
# hcl provides similar interface as TVM for the conversion between numpy arrays and
# hcl Tensor
hcl_train_images = tvm.nd.array(train_images)
_diff = tvm.nd.array(numpy.zeros((10, 1800)).astype("int64"))
_dist = tvm.nd.array(numpy.zeros((10, 1800)).astype("int32"))
result = tvm.nd.array(numpy.zeros((10, 3)).astype("int32"))

m(test_image, hcl_train_images, _diff, _dist, result)
#print _dist.asnumpy()
print result.asnumpy()
print test_label
