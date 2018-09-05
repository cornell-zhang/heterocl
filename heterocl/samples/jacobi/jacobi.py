#!/usr/bin/python
import heterocl as hcl

dtype = hcl.Float()

input_image = hcl.placeholder((480, 640), name="input", dtype=dtype)
output_image = hcl.placeholder((480, 640), name="output", dtype=dtype)

def jacobi(input_image, output_image):
  def jacobi_kernel(y, x):
    return (input_image[y, x-1] + input_image[y-1, x] + input_image[y, x] +
            input_image[y, x+1] + input_image[y+1, x])/5

  return hcl.update(output_image, jacobi_kernel, name=output_image.name)

schedule = hcl.make_schedule([input_image, output_image], jacobi)
schedule[jacobi.output].unroll(jacobi.output.axis[1], factor=8)
print(hcl.build(schedule, [input_image, output_image], target='soda'))
