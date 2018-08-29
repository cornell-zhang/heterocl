import heterocl as hcl
from math import sqrt

hcl.config.init_dtype = hcl.Float()

input_image = hcl.placeholder((480, 640), name = "input")
output_image = hcl.placeholder((480, 640), name = "output")

def gaussian(input_image, output_image):

  """
  Helper Functions
  """
  def kernel_f(x):
    return hcl.exp(-(x * x) / (2 * 1.5 * 1.5)) / sqrt(2 * 3.14159 * 1.5)

  def kernel(x):
    return kernel_f(x) * 255 / (kernel_f(0) + kernel_f(1) * 2 + kernel_f(2) * 2 + kernel_f(3) * 2 + kernel_f(4) * 2)

  rx = hcl.reduce_axis(-4, 5, "rx")
  ry = hcl.reduce_axis(-4, 5, "ry")

  out = hcl.compute(input_image.shape, lambda x, y: hcl.sum(input_image[rx+x, ry+y] * kernel(rx) * kernel(ry), axis = [rx, ry], name = "out_reduce"), name = "out")
  U = hcl.update(output_image, lambda x, y: out[x, y])

  return U

s = hcl.make_schedule([input_image, output_image], gaussian)

print hcl.lower(s, [input_image, output_image])
