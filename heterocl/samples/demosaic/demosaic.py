import heterocl as hcl

input_image = hcl.placeholder((480, 640), name = "input_image")
output_image = hcl.placeholder((480, 640, 3), name = "output_image")

def demosaic(input_image, output_image):

  nesw_neighbor = hcl.compute(input_image.shape, lambda x, y: (input_image[x-1, y] + input_image[x+1, y] + input_image[x, y-1] + input_image[x, y+1])/4)
  diag_neighbor = hcl.compute(input_image.shape, lambda x, y: (input_image[x-1, y-1] + input_image[x+1, y-1] + input_image[x-1, y+1] + input_image[x+1, y+1])/4)
  v_neighbor = hcl.compute(input_image.shape, lambda x, y: (input_image[x, y-1] + input_image[x, y+1])/2)
  h_neighbor = hcl.compute(input_image.shape, lambda x, y: (input_image[x-1, y] + input_image[x+1, y])/2)

  green = hcl.compute(input_image.shape, lambda x, y: hcl.select(y%2 == 0,
    hcl.select(x%2 == 0, nesw_neighbor[x, y], input_image[x, y]),
    hcl.select(x%2 == 0, input_image[x, y], nesw_neighbor[x, y])))

  red = hcl.compute(input_image.shape, lambda x, y: hcl.select(y%2 == 0,
    hcl.select(x%2 == 0, input_image[x, y], h_neighbor[x, y]),
    hcl.select(x%2 == 0, v_neighbor[x, y], diag_neighbor[x, y])))

  blue = hcl.compute(input_image.shape, lambda x, y: hcl.select(y%2 == 0,
    hcl.select(x%2 == 0, diag_neighbor[x, y], v_neighbor[x, y]),
    hcl.select(x%2 == 0, h_neighbor[x, y], input_image[x, y])))

  demos = hcl.compute(output_image.shape, lambda x, y, c: hcl.select(c == 0, red[x, y], hcl.select(c == 1, green[x, y], blue[x, y])))

  low_pass_y = hcl.compute(demos.shape, lambda x, y, c: (demos[x, y, c] + demos[x+1, y, c])/2)
  low_pass_x = hcl.compute(demos.shape, lambda x, y, c: (low_pass_y[x, y, c] + low_pass_y[x, y+1, c])/2)

  U = hcl.update(output_image, lambda x, y, c: low_pass_x[x, y, c])

  return U

s = hcl.make_schedule([input_image, output_image], demosaic)

print hcl.lower(s, [input_image, output_image])

