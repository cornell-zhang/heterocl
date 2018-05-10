import heterocl as hcl
import numpy as np
import tvm

def top():

  def pad(I, h, w, c):
    ih, iw, ic = I.shape
    with hcl.CodeBuilder() as cb:
      out = hcl.local(0)
      with cb._if(tvm.any(h == 0, h == ih + 1)):
        out[0] = 0
      with cb._else():
        with cb._if(tvm.any(w == 0, w == iw + 1)):
          out[0] = 0
        with cb._else():
          out[0] = I[h-1][w-1][c]
      return out[0]


  def conv2D(I, W):
    i_height, i_width, i_channel = I.shape
    w_height, w_width, w_in, w_out = W.shape
    rc = hcl.reduce_axis((0, i_channel), name="rc")
    rx = hcl.reduce_axis((0, w_height), name="rx")
    ry = hcl.reduce_axis((0, w_width), name="ry")
    PI = hcl.compute((i_height+4, i_width+4, i_channel), [I],
        lambda h, w, c: pad(I, h, w, c), name="PI")
    return hcl.compute((i_height, i_width, w_out), [PI, W],
        lambda ix, iy, wo: hcl.sum(
          PI[ix + rx, iy + ry, rc] * W[rx, ry, rc, wo],
          axis = [rc, rx, ry]))

  def maxpool2D(I):
    i_height, i_width, i_channel = I.shape
    def find_max(a, b, c, d):
      with hcl.CodeBuilder() as cb:
        _max1 = hcl.local(tvm.select(a > b, a, b), name = "m1")
        _max2 = hcl.local(tvm.select(c > d, c, d), name = "m2")
        _max = hcl.local(tvm.select(_max1 > _max2, _max1, _max2), name = "m")
        return _max[0]

    return hcl.compute((i_height/2, i_width/2, i_channel), [I],
        lambda h, w, c: find_max(I[h*2, w*2, c], I[h*2+1, w*2+1, c], I[h*2, w*2+1, c], I[h*2+1, w*2, c]))

  def reshape(I, O, w, h, c):
    i_height, i_width, i_channel = I.shape
    with hcl.CodeBuilder() as cb:
      index = hcl.local(w + h * i_width + c * i_height * i_width)
      O[index] = I[w, h, c]

  def dense(I, W):
    i_len = I.shape[0]
    w_in, w_out = W.shape
    assert i_len == w_in
    r_in = hcl.reduce_axis((0, w_in), name = "r_in")
    return hcl.compute((w_out,), [I, W], lambda o: hcl.sum(I[r_in] * W[r_in, o], axis = r_in))

  input_image = hcl.placeholder((28, 28, 1), name = "input")
  weight_1 = hcl.placeholder((5, 5, 1, 25), name = "w1")
  weight_2 = hcl.placeholder((5, 5, 25, 50), name = "w2")
  weight_3 = hcl.placeholder((2450, 500), name = "w3")
  weight_4 = hcl.placeholder((500, 10), name = "w4")

  out_1 = conv2D(input_image, weight_1)
  ap_1 = maxpool2D(out_1)

  out_2 = conv2D(ap_1, weight_2)
  ap_2 = maxpool2D(out_2)

  dense_in = hcl.compute((7*7*50,), [ap_2],
      lambda x: ap_2[x%7][(x%49)/7][x/49], name = "dense_in")

  d_1 = dense(dense_in, weight_3)
  d_2 = dense(d_1, weight_4)

  s = hcl.create_schedule(d_2)

  print tvm.lower(s, [input_image.tensor, weight_1.tensor, weight_2.tensor, weight_3.tensor, weight_4.tensor], simple_mode = True)

top()



