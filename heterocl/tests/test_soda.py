import math
import unittest

import heterocl as hcl
import tvm

def blur_func_gen():
  img_i = hcl.placeholder((2335, 235), "img_i", dtype=hcl.UInt(16))
  def blur_x(y, x):
    return (img_i[y, x-1] + img_i[y, x] + img_i[y, x+1])/3
  def blur_y(y, x):
    return (img_t[y-1, x] + img_t[y, x] + img_t[y+1, x])/3
  img_t = hcl.compute((2335, 233), blur_x, "img_t", dtype=hcl.UInt(16))
  img_o = hcl.compute((2333, 233), blur_y, "img_o", dtype=hcl.UInt(16))
  hcl_schedule = hcl.create_schedule(img_o)
  # Effective unroll factor will be the product of the unroll factors of all
  # nested loops. Recommanded practice is to unroll the innermost loop only.
  # If there are > 1 stencil stages, all unroll factors must be consistent.
  hcl_schedule[img_t].unroll(img_t.axis[1], factor=8)
  hcl_schedule[img_o].unroll(img_o.axis[1], factor=8)
  return hcl.build(hcl_schedule, [img_i, img_o], target='soda')

def gaussian_func_gen(input_image, output_image):
  def kernel_f(x):
    return hcl.exp(-(x * x) / (2 * 1.5 * 1.5)) / math.sqrt(2 * 3.14159 * 1.5)
  def kernel(x):
    return kernel_f(x) * 255 / (kernel_f(0) + kernel_f(1) * 2 +
                                kernel_f(2) * 2 + kernel_f(3) * 2 +
                                kernel_f(4) * 2)

  rx = hcl.reduce_axis(-1, 1, "rx")
  ry = hcl.reduce_axis(-1, 1, "ry")

  return hcl.update(output_image, lambda x, y: hcl.sum(
      input_image[rx+x, ry+y] * kernel(rx) * kernel(ry), axis=[rx, ry],
      name='reduce', dtype=hcl.Float()), name=output_image.name)

class TestSODA(unittest.TestCase):
  def setUp(self):
    self.maxDiff = None

  def test_blur(self):
    blur_func = blur_func_gen()
    self.assertMultiLineEqual(blur_func,
'''\
kernel: default_function
burst width: 512
unroll factor: 8
border: ignore
cluster: none
iterate: 1
output uint16:
  img_o(0, 0) = uint16((int32((uint18((uint17(img_t(0, -1)) + uint17(img_t(0, 0)))) + uint18(img_t(0, 1)))) / 3))
local uint16:
  img_t(0, 0) = uint16((int32((uint18((uint17(img_i(-1, 0)) + uint17(img_i(0, 0)))) + uint18(img_i(1, 0)))) / 3))
input uint16: img_i(233,)
''')

  def test_gaussian(self):
    img_i = hcl.placeholder((480, 640), name = "img_i", dtype=hcl.Float())
    img_o = hcl.placeholder((480, 640), name = "img_o", dtype=hcl.Float())

    schedule = hcl.make_schedule([img_i, img_o], gaussian_func_gen)
    schedule[gaussian_func_gen.img_o].unroll(gaussian_func_gen.img_o.axis[1],
                                             factor=8)
    self.assertMultiLineEqual(
        hcl.build(schedule, [img_i, img_o], target='soda'),
'''\
kernel: default_function
burst width: 512
unroll factor: 8
border: ignore
cluster: none
iterate: 1
output float32:
  reduce_ssa0 = 0.00000F
  reduce_ssa1 = float32(((float64(img_i(-1, -1)) * 2962.45) + float64(reduce_ssa0)))
  reduce_ssa2 = float32(((float64(img_i(0, -1)) * 3699.65) + float64(reduce_ssa1)))
  reduce_ssa3 = float32(((float64(img_i(-1, 0)) * 3699.65) + float64(reduce_ssa2)))
  reduce_ssa4 = float32(((float64(img_i(0, 0)) * 4620.30) + float64(reduce_ssa3)))
  img_o(0, 0) = reduce_ssa4
input float32: img_i(640,)
'''
        )

if __name__ == '__main__':
  unittest.main()
