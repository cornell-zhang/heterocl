#!/usr/bin/python
from math import sqrt
import operator
import sys
import numpy as np
import heterocl as hcl
import time 

def gaussian(input_image, output_image):
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
  soda_schedule[gaussian.output].stencil(unroll_factor=8)
  print(hcl.build(soda_schedule, target='soda'))

  with open("kernel.cpp", "w") as fp:
    kernel = hcl.build(soda_schedule, target='soda_xhls')
    fp.write(kernel)

  llvm_schedule = hcl.create_schedule([input_image, output_image], gaussian)
  program = hcl.build(llvm_schedule)

  data_in = hcl.asarray(np.random.random(input_image.shape), dtype=hcl.Float())
  data_out = hcl.asarray(np.zeros(output_image.shape), dtype=hcl.Float())

  start = time.perf_counter()
  program(data_in, data_out)
  latency = time.perf_counter() - start
  
  print(f"CPU execution time {latency}")

if __name__ == '__main__':
  main()
