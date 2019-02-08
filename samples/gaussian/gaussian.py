#!/usr/bin/python
from math import sqrt
import operator
import sys
import numpy as np
import heterocl as hcl
# please run python -m xxx/xxx from samples directory
from common import timer
import common

def gaussian(input_image, output_image):

  """
  Helper Functions
  """
  def kernel_f(x):
    return hcl.cast(hcl.Float(), hcl.exp(-(x * x) / (2 * 1.5 * 1.5)) / sqrt(2 * 3.14159 * 1.5))

  def kernel(x):
    return kernel_f(x) * 255 / (kernel_f(0) + kernel_f(1) * 2 + kernel_f(2) * 2 + kernel_f(3) * 2 + kernel_f(4) * 2)

  rx = hcl.reduce_axis(0, 3, "rx")
  ry = hcl.reduce_axis(0, 3, "ry")

  return hcl.update(output_image, lambda x, y: hcl.sum(
      input_image[rx+x, ry+y] * kernel(rx) * kernel(ry), axis=[rx, ry],
      name='reduce', dtype=hcl.Float()), name=output_image.name)

def main():
  hcl.config.init_dtype = hcl.Float()
  input_image = hcl.placeholder((480, 640), name = "input")
  output_image = hcl.placeholder((480, 640), name = "output")

  soda_schedule = hcl.create_schedule([input_image, output_image], gaussian)
  common.unroll_innermost(soda_schedule, gaussian.output, factor=8)
  # print(hcl.build(soda_schedule, target='soda'))
  print(hcl.build(soda_schedule, target='soda_xhls'))

  llvm_schedule = hcl.create_schedule([input_image, output_image], gaussian)
  program = hcl.build(llvm_schedule)

  data_in = hcl.asarray(np.random.random(input_image.shape), dtype=hcl.Float())
  data_out = hcl.asarray(np.zeros(output_image.shape), dtype=hcl.Float())

  params = timer.timeit(program, [data_in, data_out])
  sys.stderr.write('Throughput: %f pixel/ns\n' %
                   (params[0] * reduce(operator.mul, output_image.shape, 1)))

if __name__ == '__main__':
  main()
