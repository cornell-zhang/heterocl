#!/usr/bin/python
import heterocl as hcl

dtype = hcl.Float()

input_image = hcl.placeholder((480, 640), name="input", dtype=dtype)
output_image = hcl.placeholder((480, 640), name="output", dtype=dtype)

def seidel(input_image, output_image):
  rx = hcl.reduce_axis(-1, 2, "rx")
  ry = hcl.reduce_axis(-1, 2, "ry")

  tmp = hcl.compute(output_image.shape, lambda x, y: hcl.sum(
      input_image[x, ry+y], axis=[ry], dtype=dtype)/3, dtype=dtype, name='tmp')

  return hcl.update(output_image, lambda x, y: hcl.sum(
      tmp[rx+x, y], axis=[rx], dtype=dtype)/3, name=output_image.name)

schedule = hcl.make_schedule([input_image, output_image], seidel)
schedule[seidel.tmp].unroll(seidel.tmp.axis[1], factor=8)
schedule[seidel.output].unroll(seidel.output.axis[1], factor=8)
print(hcl.build(schedule, [input_image, output_image], target='soda'))
