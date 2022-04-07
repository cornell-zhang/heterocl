#!/usr/bin/python
import operator
import sys
import numpy as np
import heterocl as hcl
import time

def jacobi(input_image, output_image):
  def jacobi_kernel(y, x):
    return (input_image[y+1, x-1] + input_image[y, x] + input_image[y+1, x] +
            input_image[y+1, x+1] + input_image[y+2, x])/5

  return hcl.update(output_image, jacobi_kernel, name=output_image.name)

def main():
  dtype = hcl.Float()
  input_image = hcl.placeholder((480, 640), name="input", dtype=dtype)
  output_image = hcl.placeholder((480, 640), name="output", dtype=dtype)

  soda_schedule = hcl.create_schedule([input_image, output_image], jacobi)
  soda_schedule[jacobi.output].stencil(unroll_factor=8)
  print(hcl.build(soda_schedule, target='soda'))
  print(hcl.build(soda_schedule, target='soda_xhls'))

  with open("kernel.cpp", "w") as fp:
    kernel = hcl.build(soda_schedule, target='soda_xhls')
    fp.write(kernel)

  llvm_schedule = hcl.create_schedule([input_image, output_image], jacobi)
  program = hcl.build(llvm_schedule)

  data_in = hcl.asarray(np.random.random(input_image.shape), dtype=hcl.Float())
  data_out = hcl.asarray(np.zeros(output_image.shape), dtype=hcl.Float())

  start = time.perf_counter()
  program(data_in, data_out)
  latency = time.perf_counter() - start
  
  print(f"CPU execution time {latency}")

if __name__ == '__main__':
  main()
