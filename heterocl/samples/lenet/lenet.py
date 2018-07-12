import heterocl as hcl
import numpy as np
import tvm

dtype = "float32"

hcl.config.init_dtype = dtype

sum = hcl.reducer(0, lambda x, y: x + y, dtype)
max = hcl.reducer(-1, lambda x, y: tvm.make.Max(x, y), dtype)

def simplify(expr):
  return tvm.ir_pass.Simplify(expr) if isinstance(expr, tvm.expr.Expr) else expr


def equal_const_int(expr, value):
  if isinstance(expr, int):
    return expr == value
  if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
    expr = tvm.ir_pass.Simplify(expr)
  if not isinstance(expr, (tvm.expr.IntImm, tvm.expr.UIntImm)):
    return False
  return expr.value == value


def tanh_2d(x):
  return hcl.compute(x.shape, lambda a, b: tvm.tanh(x[a, b]))


def tanh_4d(x):
  return hcl.compute(x.shape, lambda a, b, c, d: tvm.tanh(x[a, b, c, d]))


def softmax(x):
  assert len(x.shape) == 2, "only support 2-dim softmax"
  m, n = x.shape
  k = hcl.reduce_axis(0, n)
  max_elem = hcl.compute((m, ), lambda i: max(x[i, k], axis=k))
  k = hcl.reduce_axis(0, n)
  expsum = hcl.compute(
    (m, ), lambda i: sum(tvm.exp(x[i, k] - max_elem[i]), axis=k))
  return hcl.compute(
    x.shape, lambda i, j: tvm.exp(x[i, j] - max_elem[i]) / expsum[i])


def flatten(data):
  ishape = data.shape
  dim = 1
  for i in range(1, len(ishape)):
    dim = dim * ishape[i]
  oshape = (ishape[0], dim)

  def unwrap(idx, shape):
    index = []
    for s in reversed(shape):
      index.append(idx % s)
      idx = idx / s
    return list(reversed(index))

  return hcl.compute(oshape, lambda i, j: data[tuple([i] + unwrap(j, ishape[1:]))])


def pad(data, pad_before, pad_after=None, pad_value=0.0):
  n = len(data.shape)
  pad_after = pad_after if pad_after else pad_before
  out_shape = tuple(
    tvm.ir_pass.Simplify(
      (data.shape[i] + tvm.const(pad_before[i]) + tvm.const(pad_after[i]))) for i in range(n))
  def _pad(*indices):
    not_zero = []
    index_tuple = []
    for i in range(n):
      if equal_const_int(pad_before[i], 0) and equal_const_int(pad_after[i], 0):
        index_tuple.append(indices[i])
      else:
        index_tuple.append(indices[i] - pad_before[i])
        not_zero.append(indices[i] >= pad_before[i])
        not_zero.append(indices[i] < data.shape[i] + pad_before[i])
    if not_zero:
      not_zero = tvm.all(*not_zero)
      return tvm.select(not_zero, data[tuple(index_tuple)], pad_value)
    return data[tuple(index_tuple)]
  return hcl.compute(out_shape, _pad, name='pad')


def conv2d_nchw(Input, Filter, stride=[1,1], padding=[[0,0],[0,0]]):
  out_dtype = Input.dtype
  batch, in_channel, in_height, in_width = Input.shape
  num_filter, channel, kernel_h, kernel_w = Filter.shape
  stride_h, stride_w = stride
  [pad_top, pad_left], [pad_down, pad_right] = padding
  # compute the output shape
  out_channel = num_filter
  out_height = simplify((in_height - kernel_h + pad_top + pad_down) // stride_h + 1)
  out_width = simplify((in_width - kernel_w + pad_left + pad_right) // stride_w + 1)
  # compute graph
  pad_before = [0, 0, pad_top, pad_left]
  pad_after = [0, 0, pad_down, pad_right]
  if padding != [[0,0],[0,0]]:
    Input = pad(Input, pad_before, pad_after)
  rc = hcl.reduce_axis(0, in_channel)
  ry = hcl.reduce_axis(0, kernel_h)
  rx = hcl.reduce_axis(0, kernel_w)

  return hcl.compute(
    (batch, out_channel, out_height, out_width),
      lambda nn, ff, yy, xx: sum(
        Input[nn, rc, yy * stride_h + ry, xx * stride_w + rx] *
        Filter[ff, rc, ry, rx],
        axis=[rc, ry, rx]))


def dense(data, weight, bias=None):
  assert len(data.shape) == 2 and len(weight.shape) == 2, \
    "only support 2-dim dense"
  if bias is not None:
    assert len(bias.shape) == 1
  batch, in_dim = data.shape
  out_dim, _ = weight.shape
  k = hcl.reduce_axis(0, in_dim)
  matmul = hcl.compute((batch, out_dim), \
                        lambda i, j: sum(data[i, k] * weight[j, k], axis=k))
  if bias is not None:
    matmul = hcl.compute((batch, out_dim), \
                          lambda i, j: matmul[i, j] + bias[j])
  return matmul


def max_pool(data, kernel, stride, padding=[[0,0],[0,0]]):
  assert len(data.shape) == 4, "only support 4-dim pooling"
  assert len(stride) == 2, "only support 2-dim stride"
  kernel_height, kernel_width = kernel
  stride_height, stride_width = stride
  batch, channel, height, width = data.shape
  [pad_top, pad_left], [pad_down, pad_right] = padding
  pad_before = [0, 0, pad_top, pad_left]
  pad_after = [0, 0, pad_down, pad_right]
  if padding != [[0,0],[0,0]]:
    data = pad(data, pad_before, pad_after, pad_value=tvm.min_value("float32"))
  out_height = simplify((height - kernel_height + pad_top + pad_down) // stride_height + 1)
  out_width = simplify((width - kernel_width + pad_left + pad_right) // stride_width + 1)
  dheight = hcl.reduce_axis(0, kernel_height)
  dwidth = hcl.reduce_axis(0, kernel_width)

  return hcl.compute(
    (batch, channel, out_height, out_width),
    lambda i, c, h, w:
    max(data[i, c, h*stride_height+dheight, w*stride_width+dwidth], axis=[dheight, dwidth]))

batch_size = 1000

def build_lenet(qtype1, qtype2):
  input_image = hcl.placeholder((batch_size, 1, 28, 28), name = "input_image")
  weight_conv1 = hcl.placeholder((20, 1, 5, 5), name = "weight_conv1")
  weight_conv2 = hcl.placeholder((50, 20, 5, 5), name = "weight_conv2")
  weight_fc1 = hcl.placeholder((500, 800), name = "weight_fc1")
  weight_fc2 = hcl.placeholder((10, 500), name = "weight_fc2")
  # first conv
  conv1 = conv2d_nchw(input_image, weight_conv1)
  tanh1 = tanh_4d(conv1)
  pool1 = max_pool(tanh1, kernel=(2,2), stride=(2,2))
  # second conv
  conv2 = conv2d_nchw(pool1, weight_conv2)
  tanh2 = tanh_4d(conv2)
  pool2 = max_pool(tanh2, kernel=(2,2), stride=(2,2))
  # first fc
  flat = flatten(pool2)
  fc1 = dense(flat, weight_fc1)
  tanh3 = tanh_2d(fc1)
  # second fc
  fc2 =  dense(tanh3, weight_fc2)
  # loss
  lenet = softmax(fc2)

  hcl.quantize([weight_conv1, weight_conv2, weight_fc1, weight_fc2], qtype1)
  hcl.quantize([tanh1, tanh2, tanh3], qtype2)
  # create schedule
  s = hcl.create_schedule(lenet)
  #print hcl.lower(s, [input_image, weight_conv1, weight_conv2, weight_fc1, weight_fc2, lenet])
  # build module
  f = hcl.build(s, [input_image, weight_conv1, weight_conv2, weight_fc1, weight_fc2, lenet])

  return f

import mxnet as mx
# download pretrained lenet model
mx.gluon.utils.download('https://gist.githubusercontent.com/Huyuwei/dc00ce83f537914c64a204133d23b019/raw/79af41e7c8ba9120ea7f35fb1d0484b65bccd54f/lenet-0010.params')
mx.gluon.utils.download('https://gist.githubusercontent.com/Huyuwei/dc00ce83f537914c64a204133d23b019/raw/79af41e7c8ba9120ea7f35fb1d0484b65bccd54f/lenet-symbol.json')
sym, arg_params, aux_params = mx.model.load_checkpoint('lenet', 10)
# get weights
weight_conv1_np = arg_params['convolution0_weight'].asnumpy()
weight_conv2_np = arg_params['convolution1_weight'].asnumpy()
weight_fc1_np = arg_params['fullyconnected0_weight'].asnumpy()
weight_fc2_np = arg_params['fullyconnected1_weight'].asnumpy()

# run and calculate test accuracy
qtype1 = hcl.Fixed(16, 14)
qtype2 = hcl.Fixed(16, 14)
correct_sum = 0
mnist = mx.test_utils.get_mnist()
f = build_lenet(qtype1, qtype2)

# convert weights from numpy to hcl
weight_conv1_hcl = hcl.asarray(weight_conv1_np, dtype = qtype1)
weight_conv2_hcl = hcl.asarray(weight_conv2_np, dtype = qtype1)
weight_fc1_hcl = hcl.asarray(weight_fc1_np, dtype = qtype1)
weight_fc2_hcl = hcl.asarray(weight_fc2_np, dtype = qtype1)

for i in range(10000 // batch_size):
  label = mnist['test_label'][i*batch_size:(i+1)*batch_size]
  input_image_np = mnist['test_data'][i*batch_size:(i+1)*batch_size]
  input_image_hcl = hcl.asarray(input_image_np)
  output_hcl = hcl.asarray(np.zeros((batch_size,10)))
  f(input_image_hcl, weight_conv1_hcl, weight_conv2_hcl, weight_fc1_hcl, weight_fc2_hcl, output_hcl)
  prediction = np.argmax(output_hcl.asnumpy(), axis=1)
  correct_sum += np.sum(np.equal(prediction, label))

print(str(qtype1) + ", " + str(qtype2) + ": Accuracy over 10000 test images is: {}".format(correct_sum / 10000.))
