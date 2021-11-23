import heterocl as hcl
from math import sqrt

hcl.config.init_dtype = hcl.Float()

input_image = hcl.placeholder((480, 640, 3), name = "input")
output_image = hcl.placeholder((480, 640, 3), name = "output")

def unsharp(input_image, output_image):

  """
  Helper Functions
  """
  def clamp(val, min_, max_):
    local = hcl.scalar(val)
    with hcl.if_(val < min_):
      local[0] = min_
    with hcl.elif_(val > max_):
      local[0] = max_
    return local[0]

  def clamp2D(tensor, min_, max_):
    return hcl.compute(tensor.shape, lambda x, y: clamp(tensor[x, y], min_, max_), name = "clamped_" + tensor.name)

  def clamp3D(tensor, min_, max_):
    return hcl.compute(tensor.shape, lambda x, y, c: clamp(tensor[x, y, c], min_, max_), name = "clamped_" + tensor.name)

  def kernel_f(x):
    return hcl.exp(-(x * x) / (2 * 1.5 * 1.5)) / sqrt(2 * 3.14159 * 1.5)

  def kernel(x):
    return kernel_f(x) * 255 / (kernel_f(0) + kernel_f(1) * 2 + kernel_f(2) * 2 + kernel_f(3) * 2 + kernel_f(4) * 2)

  rx = hcl.reduce_axis(-4, 5, "rx")
  ry = hcl.reduce_axis(-4, 5, "ry")
  my = hcl.reduce_axis(0, 640, "my")

  gray = hcl.compute((480, 640), lambda x, y: (input_image[x, y, 0] * 77 + input_image[x, y, 1] * 150 + input_image[x, y, 2] * 29) >> 8, name = "gray")
  blur = hcl.compute(gray.shape, lambda x, y: hcl.sum(gray[rx+x, ry+y] * kernel(rx) * kernel(ry), axis = [rx, ry]), name = "blur")
  sharpen = clamp2D(hcl.compute(gray.shape, lambda x, y: gray[x, y] * 2 - blur[x, y], name = "sharpen"), 0, 255)
  ratio = clamp2D(hcl.compute(gray.shape, lambda x, y: sharpen[x, y] * 32 / hcl.max(gray[x, my], axis = my), name = "ratio"), 0, 255)
  out = clamp3D(hcl.compute(output_image.shape, lambda x, y, c: ratio[x, y] * input_image[x, y, c] >> 5, name = "out"), 0, 255)
  U = hcl.update(output_image, lambda x, y, c: out[x, y, c])

  return U

s = hcl.make_schedule([input_image, output_image], unsharp)

print hcl.lower(s, [input_image, output_image])
