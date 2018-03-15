import hcl

def popcount(num, nbits):
  out = 0
  for i in range(0, nbits):
    if x%2 == 1:
      out = out + 1
    out = out >> 1
  return out

def popcount_inline(num, nbits):
  out = 0
  for i in range(0, nbits):
    out = hcl.select(x%2, out+1, out)
    out = out >> 1
  return out

def update_knn(inputs, outputs, args):
  dist = inputs[0]
  knn_mat = outputs[0]
  for i in range(0, 10):
    for j in range(0, 100):
      max_id = 0
      for k in range(1, 3):
        max_id = k if knn_mat[i][k] > knn_mat[i][max_id] else max_id
      if dist[i][j] < knn_mat[i][max_id]:
        knn_mat[i][max_id] = dist[i][j]

""" function definitons and updates """
input_image = hcl.var(name = "input_image", dtype = "int_49")
labelval = hcl.placeholder((10, 100), name = "labelval", dtype = "int_49")
diff = hcl.compute(labelval.shape, lambda x, y: input_image ^ labelval[x][y], name = "diff")
dist = hcl.compute((10, 100), lambda x, y: popcount(diff[x][y], 49), name = "dist", inline = False)
# alternatively, you can do this
# dist = hcl.compute((10, 100), lambda x, y: popcount_inline(diff[x][y], 49), name = "dist")
knn_mat = hcl.placeholder((10, 3), name = "knn_mat", dtype = "int_6")
knn_init = hcl.update(knn_mat, lambda x, y: knn_mat[x][y] = 50, name = "knn_init")
knn_update = hcl.block(update_knn, [dist], [knn_mat], None, name = "knn_update")

""" scheduling """
s = hcl.create_schedule(knn_update)
# note that diff and dist are Tensor while knn_int and knn_update are Stage
# the axes of a Tensor are the axes of the Tensor's definition
# a.axis == a.stages[0].axis
s[diff].unroll(diff.axis[0]) # diff.axis = [x, y]
s[dist].pipeline(dist.axis[1], II = 1) # dist.axis = [x, y, i]
s[knn_init].unroll(knn_init.axis[0]) # knn_init.axis = [x, y]
s[knn_update].pipeline(knn_update.axis[1], II = 1) # knn_update.axis = [i, j, k]

""" build the module """
m = hcl.build(s, [input_image, label_val, knn_mat])

""" test the module """
# hcl provides similar interface as TVM for the conversion between numpy arrays and
# hcl Tensor
test_image = 0x1234
train_image = hcl.asarray(numpy.load(...))
result = hcl.asarray(numpy.zeros((10, 3)))
m(test_image, train_image, result)
print result.asnumpy()
