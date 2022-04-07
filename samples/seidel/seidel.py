#!/usr/bin/python
import operator
import sys
import numpy as np
import heterocl as hcl
import time

def seidel(input_image, output_image):
  dtype = hcl.Float()
  rx = hcl.reduce_axis(0, 3, "rx")
  ry = hcl.reduce_axis(0, 3, "ry")

  tmp = hcl.compute(output_image.shape, lambda x, y: hcl.sum(
      input_image[x, ry+y], axis=[ry], dtype=dtype)/3, dtype=dtype, name='tmp')

  return hcl.update(output_image, lambda x, y: hcl.sum(
      tmp[rx+x, y], axis=[rx], dtype=dtype)/3, name=output_image.name)

def main():
  dtype = hcl.Float()
  input_image = hcl.placeholder((480, 640), name="input", dtype=dtype)
  output_image = hcl.placeholder((480, 640), name="output", dtype=dtype)

  soda_schedule = hcl.create_schedule([input_image, output_image], seidel)
  soda_schedule[seidel.tmp].stencil(unroll_factor=8)
  soda_schedule[seidel.output].stencil(unroll_factor=8)

  print(hcl.build(soda_schedule, target='soda'))
  print(hcl.build(soda_schedule, target='soda_xhls'))

  with open("kernel.cpp", "w") as fp:
    kernel = hcl.build(soda_schedule, target='soda_xhls')
    fp.write(kernel)

  llvm_schedule = hcl.create_schedule([input_image, output_image], seidel)
  program = hcl.build(llvm_schedule)

  data_in = hcl.asarray(np.random.random(input_image.shape), dtype=hcl.Float())
  data_out = hcl.asarray(np.zeros(output_image.shape), dtype=hcl.Float())

  start = time.perf_counter()
  program(data_in, data_out)
  latency = time.perf_counter() - start
  
  print(f"CPU execution time {latency}")

if __name__ == '__main__':
  main()
