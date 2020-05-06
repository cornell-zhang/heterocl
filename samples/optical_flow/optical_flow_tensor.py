import heterocl as hcl
import numpy as np
import os, sys
from PIL import Image
# np.set_printoptions(threshold=sys.maxsize)

# height x width
size = (436, 1024)
height, width = size
hcl.init(hcl.Fixed(32,12))
dtype = hcl.Fixed(32,12)

# setup target using vivado 
tool = hcl.tool.sdaccel
tool.mode = "hw_emu"
os.environ["AWS_PLATFORM"] = "xilinx_vcu1525_dynamic_5_1"
target = hcl.platform.aws_f1(tool)
target.xcel.lang = "vhls"
# target = "llvm"

# load ppm image amd convert to grayscale
img0 = Image.open("datasets/current/frame1.ppm").convert("L")
img1 = Image.open("datasets/current/frame2.ppm").convert("L") 
img2 = Image.open("datasets/current/frame3.ppm").convert("L")
img3 = Image.open("datasets/current/frame4.ppm").convert("L")
img4 = Image.open("datasets/current/frame5.ppm").convert("L")

img0 = np.asarray(img0.getdata(), dtype=np.uint32).reshape(img0.size[1], img0.size[0]) 
img1 = np.asarray(img1.getdata(), dtype=np.uint32).reshape(img1.size[1], img1.size[0]) 
img2 = np.asarray(img2.getdata(), dtype=np.uint32).reshape(img2.size[1], img2.size[0]) 
img3 = np.asarray(img3.getdata(), dtype=np.uint32).reshape(img3.size[1], img3.size[0]) 
img4 = np.asarray(img4.getdata(), dtype=np.uint32).reshape(img4.size[1], img4.size[0]) 
imgs = [img0, img1, img2, img2, img2, img3, img4]


def optical_flow(target=target):

    image0 = hcl.placeholder((436,1024), "input_image0")
    image1 = hcl.placeholder((436,1024), "input_image1")
    image2 = hcl.placeholder((436,1024), "input_image2")
    image2_0 = hcl.placeholder((436,1024), "input_image2_0")
    image2_1 = hcl.placeholder((436,1024), "input_image2_1")
    image3 = hcl.placeholder((436,1024), "input_image3")
    image4 = hcl.placeholder((436,1024), "input_image4")
    output = hcl.placeholder((436,1024,2), "output_image")

    def kernel(img0, img1, img2, img2_0, img2_1, img3, img4, output):

       sum = hcl.reducer(0, lambda x, y: x + y, dtype)

       @hcl.def_([size, size])
       def calc_x_gradient(input_image_0, grad_x):
           g_w = hcl.copy([1, -8, 0, 8, 1], "g_w", hcl.Int())
           rx = hcl.reduce_axis(0, 5, name="rdx")
           def update(y, x):
               with hcl.if_(hcl.and_(y>=2, y<height-2, x>=2, x<width-2)):
                   grad_x[y,x] = sum(input_image_0[y, x+rx-2] * g_w[rx], axis=rx)
           hcl.mutate(size, lambda y, x: update(y, x))
           
       @hcl.def_([size, size])
       def calc_y_gradient(input_image_1, grad_y):
           g_w = hcl.copy([1, -8, 0, 8, 1], "g_w", hcl.Int())
           ry = hcl.reduce_axis(0, 5, name="rdy")
           def update(y, x):
               with hcl.if_(hcl.and_(y>=2, y<height-2, x>=2, x<width-2)):
                   grad_y[y,x] = sum(input_image_1[y+ry-2, x] * g_w[ry], axis=ry)
           hcl.mutate(size, lambda y, x: update(y, x))
           

       @hcl.def_([size, size, size, size, size, size])
       def calc_z_gradient(img0, img1, img2_0, img3, img4, grad_z):
           g_w = hcl.copy([1, -8, 0, 8, 1], "g_w", hcl.Int())
           hcl.update(grad_z, 
               lambda y, x: (img0[y,x] * g_w[0] +
                             img1[y,x] * g_w[1] +
                             img2_0[y,x] * g_w[2] +
                             img3[y,x] * g_w[3] +
                             img4[y,x] * g_w[4]) / 12.0)

       # averaging gradients in y dim
       @hcl.def_([size, size])
       def grad_weight_y_0(grad_x, y_filt_0):
           g_f = hcl.copy([0.0755, 0.133, 0.1869, 0.2903, \
                           0.1869, 0.133, 0.0755], "g_f", hcl.Float())
           rd = hcl.reduce_axis(0, 7, name="rdx")
           def acc(y, x):
               with hcl.if_(hcl.and_(y>=3, y<height-3)):
                   y_filt_0[y, x] = sum(grad_x[y+rd-3,x] * g_f[rd], axis=rd)
           hcl.mutate(y_filt_0.shape, lambda y, x: acc(y, x))

       # averaging gradients in y dim
       @hcl.def_([size, size])
       def grad_weight_y_1(grad_y, y_filt_1):
           g_f = hcl.copy([0.0755, 0.133, 0.1869, 0.2903, \
                           0.1869, 0.133, 0.0755], "g_f", hcl.Float())
           rd = hcl.reduce_axis(0, 7, name="rdx")
           def acc(y, x):
               with hcl.if_(hcl.and_(y>=3, y<height-3)):
                   y_filt_1[y, x] = sum(grad_y[y+rd-3,x] * g_f[rd], axis=rd)
           hcl.mutate(y_filt_1.shape, lambda y, x: acc(y, x))

       # averaging gradients in y dim
       @hcl.def_([size, size])
       def grad_weight_y_2(grad_z, y_filt_2):
           g_f = hcl.copy([0.0755, 0.133, 0.1869, 0.2903, \
                           0.1869, 0.133, 0.0755], "g_f", hcl.Float())
           rd = hcl.reduce_axis(0, 7, name="rdx")
           def acc(y, x):
               with hcl.if_(hcl.and_(y>=3, y<height-3)):
                   y_filt_2[y, x] = sum(grad_z[y+rd-3,x] * g_f[rd], axis=rd)
           hcl.mutate(y_filt_2.shape, lambda y, x: acc(y, x))

       @hcl.def_([(436,1024), (436,1024)])
       def grad_weight_x_0(y_filt_0, filt_grad_0):
           g_f = hcl.copy([0.0755, 0.133, 0.1869, 0.2903, \
                           0.1869, 0.133, 0.0755], "g_f", hcl.Float())
           rd = hcl.reduce_axis(0, 7, name="rdx")
           def acc(y, x):
               with hcl.if_(hcl.and_(x>=3, x<width-3)):
                   filt_grad_0[y, x] = sum(y_filt_0[y, x+rd-3] * g_f[rd], axis=rd)
           hcl.mutate(filt_grad_0.shape, lambda y, x: acc(y, x))


       @hcl.def_([(436,1024), (436,1024)])
       def grad_weight_x_1(y_filt_1, filt_grad_1):
           g_f = hcl.copy([0.0755, 0.133, 0.1869, 0.2903, \
                           0.1869, 0.133, 0.0755], "g_f", hcl.Float())
           rd = hcl.reduce_axis(0, 7, name="rdx")
           def acc(y, x):
               with hcl.if_(hcl.and_(x>=3, x<width-3)):
                   filt_grad_1[y, x] = sum(y_filt_1[y, x+rd-3] * g_f[rd], axis=rd)
           hcl.mutate(filt_grad_1.shape, lambda y, x: acc(y, x))


       @hcl.def_([(436,1024), (436,1024)])
       def grad_weight_x_2(y_filt_2, filt_grad_2):
           g_f = hcl.copy([0.0755, 0.133, 0.1869, 0.2903, \
                           0.1869, 0.133, 0.0755], "g_f", hcl.Float())
           rd = hcl.reduce_axis(0, 7, name="rdx")
           def acc(y, x):
               with hcl.if_(hcl.and_(x>=3, x<width-3)):
                   filt_grad_2[y, x] = sum(y_filt_2[y, x+rd-3] * g_f[rd], axis=rd)
           hcl.mutate(filt_grad_2.shape, lambda y, x: acc(y, x))


       @hcl.def_([size, size, size, size, size, size, size, size, size])
       def outer_product(filt_grad_0, filt_grad_1, filt_grad_2, 
                         out_product_0, out_product_1, out_product_2,
                         out_product_3, out_product_4, out_product_5):
           def update(y, x):
               a = hcl.scalar(filt_grad_0[y,x], "a") 
               b = hcl.scalar(filt_grad_1[y,x], "b") 
               c = hcl.scalar(filt_grad_2[y,x], "c") 
               out_product_0[y,x] = a.v * a.v
               out_product_1[y,x] = b.v * b.v
               out_product_2[y,x] = c.v * c.v
               out_product_3[y,x] = a.v * b.v
               out_product_4[y,x] = a.v * c.v
               out_product_5[y,x] = b.v * c.v
           hcl.mutate(size, lambda y, x: update(y, x))

       @hcl.def_([(436,1024), (436,1024)])
       def tensor_weight_x_0(tensor_y_0, tensor_0):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx_x")
           def acc(y, x):
               with hcl.if_(hcl.and_(x>=1, x<width-1)):
                   tensor_0[y, x] = sum(tensor_y_0[y,x+rd-1] * t_w[rd], axis=rd)
           hcl.mutate(tensor_0.shape, lambda y, x: acc(y, x))

       @hcl.def_([(436,1024), (436,1024)])
       def tensor_weight_x_1(tensor_y_1, tensor_1):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx_x")
           def acc(y, x):
               with hcl.if_(hcl.and_(x>=1, x<width-1)):
                   tensor_1[y, x] = sum(tensor_y_1[y,x+rd-1] * t_w[rd], axis=rd)
           hcl.mutate(tensor_1.shape, lambda y, x: acc(y, x))

       @hcl.def_([(436,1024), (436,1024)])
       def tensor_weight_x_2(tensor_y_2, tensor_2):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx_x")
           def acc(y, x):
               with hcl.if_(hcl.and_(x>=1, x<width-1)):
                   tensor_2[y, x] = sum(tensor_y_2[y,x+rd-1] * t_w[rd], axis=rd)
           hcl.mutate(tensor_2.shape, lambda y, x: acc(y, x))

       @hcl.def_([(436,1024), (436,1024)])
       def tensor_weight_x_3(tensor_y_3, tensor_3):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx_x")
           def acc(y, x):
               with hcl.if_(hcl.and_(x>=1, x<width-1)):
                   tensor_3[y, x] = sum(tensor_y_3[y,x+rd-1] * t_w[rd], axis=rd)
           hcl.mutate(tensor_3.shape, lambda y, x: acc(y, x))

       @hcl.def_([(436,1024), (436,1024)])
       def tensor_weight_x_4(tensor_y_4, tensor_4):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx_x")
           def acc(y, x):
               with hcl.if_(hcl.and_(x>=1, x<width-1)):
                   tensor_4[y, x] = sum(tensor_y_4[y,x+rd-1] * t_w[rd], axis=rd)
           hcl.mutate(tensor_4.shape, lambda y, x: acc(y, x))

       @hcl.def_([(436,1024), (436,1024)])
       def tensor_weight_x_5(tensor_y_5, tensor_5):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx_x")
           def acc(y, x):
               with hcl.if_(hcl.and_(x>=1, x<width-1)):
                   tensor_5[y, x] = sum(tensor_y_5[y,x+rd-1] * t_w[rd], axis=rd)
           hcl.mutate(tensor_5.shape, lambda y, x: acc(y, x))

       @hcl.def_([(436,1024), (436,1024)])
       def tensor_weight_y_0(out_product_0, tensor_y_0):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx_y")
           def acc(y, x):
               with hcl.if_(hcl.and_(y>=1, y<height-1)):
                   tensor_y_0[y, x] = sum(out_product_0[y+rd-1,x] * t_w[rd], axis=rd)
           hcl.mutate(tensor_y_0.shape, lambda y, x: acc(y, x))

       @hcl.def_([(436,1024), (436,1024)])
       def tensor_weight_y_1(out_product_1, tensor_y_1):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx_y")
           def acc(y, x):
               with hcl.if_(hcl.and_(y>=1, y<height-1)):
                   tensor_y_1[y, x] = sum(out_product_1[y+rd-1,x] * t_w[rd], axis=rd)
           hcl.mutate(tensor_y_1.shape, lambda y, x: acc(y, x))

       @hcl.def_([(436,1024), (436,1024)])
       def tensor_weight_y_2(out_product_2, tensor_y_2):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx_y")
           def acc(y, x):
               with hcl.if_(hcl.and_(y>=1, y<height-1)):
                   tensor_y_2[y, x] = sum(out_product_2[y+rd-1,x] * t_w[rd], axis=rd)
           hcl.mutate(tensor_y_2.shape, lambda y, x: acc(y, x))

       @hcl.def_([(436,1024), (436,1024)])
       def tensor_weight_y_3(out_product_3, tensor_y_3):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx_y")
           def acc(y, x):
               with hcl.if_(hcl.and_(y>=1, y<height-1)):
                   tensor_y_3[y, x] = sum(out_product_3[y+rd-1,x] * t_w[rd], axis=rd)
           hcl.mutate(tensor_y_3.shape, lambda y, x: acc(y, x))

       @hcl.def_([(436,1024), (436,1024)])
       def tensor_weight_y_4(out_product_4, tensor_y_4):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx_y")
           def acc(y, x):
               with hcl.if_(hcl.and_(y>=1, y<height-1)):
                   tensor_y_4[y, x] = sum(out_product_4[y+rd-1,x] * t_w[rd], axis=rd)
           hcl.mutate(tensor_y_4.shape, lambda y, x: acc(y, x))

       @hcl.def_([(436,1024), (436,1024)])
       def tensor_weight_y_5(out_product_5, tensor_y_5):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx_y")
           def acc(y, x):
               with hcl.if_(hcl.and_(y>=1, y<height-1)):
                   tensor_y_5[y, x] = sum(out_product_5[y+rd-1,x] * t_w[rd], axis=rd)
           hcl.mutate(tensor_y_5.shape, lambda y, x: acc(y, x))

       @hcl.def_([size, size, size, size, size, size, (436,1024,2)])
       def flow_calc(tensor_0, tensor_1, tensor_2, tensor_3,
                     tensor_4, tensor_5, output):
           with hcl.for_(0, height, name="y") as y:
             with hcl.for_(0, width, name="x") as x:
               with hcl.if_(hcl.and_(y>=2, y<height-2, x>=2, x<width-2)):
                 t0 = hcl.scalar(tensor_0[y,x], "t0")
                 t1 = hcl.scalar(tensor_1[y,x], "t1")
                 t2 = hcl.scalar(tensor_2[y,x], "t2")
                 t3 = hcl.scalar(tensor_3[y,x], "t3")
                 t4 = hcl.scalar(tensor_4[y,x], "t4")
                 t5 = hcl.scalar(tensor_5[y,x], "t5")
                 s0 = hcl.scalar(t1.v*t2.v-t4.v*t4.v, "denom")
                 output[y,x,0] = (t5.v*t3.v-t4.v*t1.v) / s0.v
                 output[y,x,1] = (t4.v*t3.v-t5.v*t0.v) / s0.v

       init = lambda *args: 0
       grad_x = hcl.compute(size, init, name="grad_x")
       grad_y = hcl.compute(size, init, name="grad_y")
       grad_z = hcl.compute(size, init, name="grad_z")

       # decompose into separate channels 
       y_filt_0      = hcl.compute(size, init, name="y_filt_0")
       y_filt_1      = hcl.compute(size, init, name="y_filt_1")
       y_filt_2      = hcl.compute(size, init, name="y_filt_2")

       filt_grad_0   = hcl.compute(size, init, name="filt_grad_0")
       filt_grad_1   = hcl.compute(size, init, name="filt_grad_1")
       filt_grad_2   = hcl.compute(size, init, name="filt_grad_2")

       out_product_0 = hcl.compute((436,1024), init, name="out_product_0")
       out_product_1 = hcl.compute((436,1024), init, name="out_product_1")
       out_product_2 = hcl.compute((436,1024), init, name="out_product_2")
       out_product_3 = hcl.compute((436,1024), init, name="out_product_3")
       out_product_4 = hcl.compute((436,1024), init, name="out_product_4")
       out_product_5 = hcl.compute((436,1024), init, name="out_product_5")

       tensor_y_0 = hcl.compute((436,1024), init, name="tensor_y_0")
       tensor_y_1 = hcl.compute((436,1024), init, name="tensor_y_1")
       tensor_y_2 = hcl.compute((436,1024), init, name="tensor_y_2")
       tensor_y_3 = hcl.compute((436,1024), init, name="tensor_y_3")
       tensor_y_4 = hcl.compute((436,1024), init, name="tensor_y_4")
       tensor_y_5 = hcl.compute((436,1024), init, name="tensor_y_5")

       tensor_0   = hcl.compute((436,1024), init, name="tensor_0")
       tensor_1   = hcl.compute((436,1024), init, name="tensor_1")
       tensor_2   = hcl.compute((436,1024), init, name="tensor_2")
       tensor_3   = hcl.compute((436,1024), init, name="tensor_3")
       tensor_4   = hcl.compute((436,1024), init, name="tensor_4")
       tensor_5   = hcl.compute((436,1024), init, name="tensor_5")

       calc_x_gradient(image2_0, grad_x)
       calc_y_gradient(image2_1, grad_y)
       calc_z_gradient(image0, image1, image2, image3, image4, grad_z)

       # calc on 3 different dimension for y_filt 
       grad_weight_y_0(grad_x, y_filt_0)
       grad_weight_y_1(grad_y, y_filt_1)
       grad_weight_y_2(grad_z, y_filt_2)

       # calc on 3 different dimension for filt_grad 
       grad_weight_x_0(y_filt_0, filt_grad_0)
       grad_weight_x_1(y_filt_1, filt_grad_1)
       grad_weight_x_2(y_filt_2, filt_grad_2)

       # calc on 6 different dimension for out_product 
       outer_product(filt_grad_0, filt_grad_1, filt_grad_2, 
                     out_product_0, out_product_1, out_product_2,
                     out_product_3, out_product_4, out_product_5)

       # calc on 6 dim for tensor_y (1-input 1-output for data resue)
       tensor_weight_y_0(out_product_0, tensor_y_0)
       tensor_weight_y_1(out_product_1, tensor_y_1)
       tensor_weight_y_2(out_product_2, tensor_y_2)
       tensor_weight_y_3(out_product_3, tensor_y_3)
       tensor_weight_y_4(out_product_4, tensor_y_4)
       tensor_weight_y_5(out_product_5, tensor_y_5)

       # calc on 6 dim for tensor (1-input 1-output for data resue)
       tensor_weight_x_0(tensor_y_0, tensor_0)
       tensor_weight_x_1(tensor_y_1, tensor_1)
       tensor_weight_x_2(tensor_y_2, tensor_2)
       tensor_weight_x_3(tensor_y_3, tensor_3)
       tensor_weight_x_4(tensor_y_4, tensor_4)
       tensor_weight_x_5(tensor_y_5, tensor_5)

       flow_calc(tensor_0, tensor_1, tensor_2, tensor_3, 
                 tensor_4, tensor_5, output)

    s = hcl.create_schedule([image0, image1, image2, image2_0, image2_1, 
                             image3, image4, output], kernel)

    if target != "llvm":

      # transmit packed data to device 
      s.to([image4, image3, image2, image2_1, image2_0, image1, image0], target.xcel)
      s.to(output, target.host)

      k_grad_x    = kernel.calc_x_gradient
      k_grad_y    = kernel.calc_y_gradient
      k_grad_z    = kernel.calc_z_gradient

      k_grad_weight_y_0    = kernel.grad_weight_y_0
      k_grad_weight_y_1    = kernel.grad_weight_y_1
      k_grad_weight_y_2    = kernel.grad_weight_y_2

      k_grad_weight_x_0    = kernel.grad_weight_x_0
      k_grad_weight_x_1    = kernel.grad_weight_x_1
      k_grad_weight_x_2    = kernel.grad_weight_x_2

      k_outer_product      = kernel.outer_product

      k_tensor_x_0  = kernel.tensor_weight_x_0
      k_tensor_x_1  = kernel.tensor_weight_x_1
      k_tensor_x_2  = kernel.tensor_weight_x_2
      k_tensor_x_3  = kernel.tensor_weight_x_3
      k_tensor_x_4  = kernel.tensor_weight_x_4
      k_tensor_x_5  = kernel.tensor_weight_x_5

      k_tensor_y_0  = kernel.tensor_weight_y_0
      k_tensor_y_1  = kernel.tensor_weight_y_1
      k_tensor_y_2  = kernel.tensor_weight_y_2
      k_tensor_y_3  = kernel.tensor_weight_y_3
      k_tensor_y_4  = kernel.tensor_weight_y_4
      k_tensor_y_5  = kernel.tensor_weight_y_5

      k_calc_flow = kernel.flow_calc

      # reuse buffer
      rb_0 = s.reuse_at(k_grad_x.input_image_0, s[k_grad_x], k_grad_x.axis[1])
      s.partition(rb_0, dim=2)
      rb_1 = s.reuse_at(k_grad_y.input_image_1, s[k_grad_y], k_grad_y.axis[0])
      s.partition(rb_1, dim=1)

      # reuse to calculate y_filt
      rb_2 = s.reuse_at(k_grad_weight_y_0.grad_x, 
                 s[k_grad_weight_y_0], k_grad_weight_y_0.axis[0])
      s.partition(rb_2, dim=1)
      rb_3 = s.reuse_at(k_grad_weight_y_1.grad_y, 
                 s[k_grad_weight_y_1], k_grad_weight_y_1.axis[0])
      s.partition(rb_3, dim=1)
      rb_4 = s.reuse_at(k_grad_weight_y_2.grad_z, 
                 s[k_grad_weight_y_2], k_grad_weight_y_2.axis[0])
      s.partition(rb_4, dim=1)

      # reuse for filt_grad
      rb_5 = s.reuse_at(k_grad_weight_x_0.y_filt_0, 
                 s[k_grad_weight_x_0], k_grad_weight_x_0.axis[1])
      s.partition(rb_5, dim=2)
      rb_6 = s.reuse_at(k_grad_weight_x_1.y_filt_1, 
                 s[k_grad_weight_x_1], k_grad_weight_x_1.axis[1])
      s.partition(rb_6, dim=2)
      rb_7 = s.reuse_at(k_grad_weight_x_2.y_filt_2, 
                 s[k_grad_weight_x_2], k_grad_weight_x_2.axis[1])
      s.partition(rb_7, dim=2)

      # reuse for tensor_weight_y
      rb_8 = s.reuse_at(k_tensor_y_0.out_product_0, 
                 s[k_tensor_y_0], k_tensor_y_0.axis[0])
      s.partition(rb_8, dim=1)
      rb_9 = s.reuse_at(k_tensor_y_1.out_product_1, 
                 s[k_tensor_y_1], k_tensor_y_1.axis[0])
      s.partition(rb_9, dim=1)
      rb_10 = s.reuse_at(k_tensor_y_2.out_product_2, 
                 s[k_tensor_y_2], k_tensor_y_2.axis[0])
      s.partition(rb_10, dim=1)
      rb_11 = s.reuse_at(k_tensor_y_3.out_product_3, 
                 s[k_tensor_y_3], k_tensor_y_3.axis[0])
      s.partition(rb_11, dim=1)
      rb_12 = s.reuse_at(k_tensor_y_4.out_product_4, 
                 s[k_tensor_y_4], k_tensor_y_4.axis[0])
      s.partition(rb_12, dim=1)
      rb_13 = s.reuse_at(k_tensor_y_5.out_product_5, 
                 s[k_tensor_y_5], k_tensor_y_5.axis[0])
      s.partition(rb_13, dim=1)

      # reuse for tensor_weight_x
      rb_14 = s.reuse_at(k_tensor_x_0.tensor_y_0, 
                 s[k_tensor_x_0], k_tensor_x_0.axis[1])
      s.partition(rb_14, dim=2)
      rb_15 = s.reuse_at(k_tensor_x_1.tensor_y_1, 
                 s[k_tensor_x_1], k_tensor_x_1.axis[1])
      s.partition(rb_15, dim=2)
      rb_16 = s.reuse_at(k_tensor_x_2.tensor_y_2, 
                 s[k_tensor_x_2], k_tensor_x_2.axis[1])
      s.partition(rb_16, dim=2)
      rb_17 = s.reuse_at(k_tensor_x_3.tensor_y_3, 
                 s[k_tensor_x_3], k_tensor_x_3.axis[1])
      s.partition(rb_17, dim=2)
      rb_18 = s.reuse_at(k_tensor_x_4.tensor_y_4, 
                 s[k_tensor_x_4], k_tensor_x_4.axis[1])
      s.partition(rb_18, dim=2)
      rb_19 = s.reuse_at(k_tensor_x_5.tensor_y_5, 
                 s[k_tensor_x_5], k_tensor_x_5.axis[1])
      s.partition(rb_19, dim=2)

      # creat streaming channels + reuse buffer
      s.to(kernel.grad_x, 
           s[k_grad_weight_y_0], s[k_grad_x], hcl.Stream.FIFO)
      s.to(kernel.grad_y, 
           s[k_grad_weight_y_1], s[k_grad_y], hcl.Stream.FIFO)
      s.to(kernel.grad_z, 
           s[k_grad_weight_y_2], s[k_grad_z], hcl.Stream.FIFO)

      s.to(kernel.y_filt_0, 
           s[k_grad_weight_x_0], s[k_grad_weight_y_0], hcl.Stream.FIFO)
      s.to(kernel.y_filt_1, 
           s[k_grad_weight_x_1], s[k_grad_weight_y_1], hcl.Stream.FIFO)
      s.to(kernel.y_filt_2, 
           s[k_grad_weight_x_2], s[k_grad_weight_y_2], hcl.Stream.FIFO)

      s.to(kernel.filt_grad_0, 
           s[k_outer_product], s[k_grad_weight_x_0], hcl.Stream.FIFO)
      s.to(kernel.filt_grad_1, 
           s[k_outer_product], s[k_grad_weight_x_1], hcl.Stream.FIFO)
      s.to(kernel.filt_grad_2, 
           s[k_outer_product], s[k_grad_weight_x_2], hcl.Stream.FIFO)

      s.to(kernel.out_product_0, 
           s[k_tensor_y_0], s[k_outer_product], hcl.Stream.FIFO)
      s.to(kernel.out_product_1, 
           s[k_tensor_y_1], s[k_outer_product], hcl.Stream.FIFO)
      s.to(kernel.out_product_2, 
           s[k_tensor_y_2], s[k_outer_product], hcl.Stream.FIFO)
      s.to(kernel.out_product_3, 
           s[k_tensor_y_3], s[k_outer_product], hcl.Stream.FIFO)
      s.to(kernel.out_product_4, 
           s[k_tensor_y_4], s[k_outer_product], hcl.Stream.FIFO)
      s.to(kernel.out_product_5, 
           s[k_tensor_y_5], s[k_outer_product], hcl.Stream.FIFO)

      s.to(kernel.tensor_y_0, 
           s[k_tensor_x_0], s[k_tensor_y_0], hcl.Stream.FIFO)
      s.to(kernel.tensor_y_1, 
           s[k_tensor_x_1], s[k_tensor_y_1], hcl.Stream.FIFO)
      s.to(kernel.tensor_y_2, 
           s[k_tensor_x_2], s[k_tensor_y_2], hcl.Stream.FIFO)
      s.to(kernel.tensor_y_3, 
           s[k_tensor_x_3], s[k_tensor_y_3], hcl.Stream.FIFO)
      s.to(kernel.tensor_y_4, 
           s[k_tensor_x_4], s[k_tensor_y_4], hcl.Stream.FIFO)
      s.to(kernel.tensor_y_5, 
           s[k_tensor_x_5], s[k_tensor_y_5], hcl.Stream.FIFO)

      s.to(kernel.tensor_0, 
           s[k_calc_flow], s[k_tensor_x_0], hcl.Stream.FIFO)
      s.to(kernel.tensor_1, 
           s[k_calc_flow], s[k_tensor_x_1], hcl.Stream.FIFO)
      s.to(kernel.tensor_2, 
           s[k_calc_flow], s[k_tensor_x_2], hcl.Stream.FIFO)
      s.to(kernel.tensor_3, 
           s[k_calc_flow], s[k_tensor_x_3], hcl.Stream.FIFO)
      s.to(kernel.tensor_4, 
           s[k_calc_flow], s[k_tensor_x_4], hcl.Stream.FIFO)
      s.to(kernel.tensor_5, 
           s[k_calc_flow], s[k_tensor_x_5], hcl.Stream.FIFO)

      # pipeline streaming rd/wr 
      s[k_grad_x].pipeline(k_grad_x.axis[1])
      s[k_grad_y].pipeline(k_grad_y.axis[1])
      s[k_grad_z].pipeline(k_grad_z.axis[1])

      s[k_grad_weight_x_0].pipeline(k_grad_weight_x_0.axis[1])
      s[k_grad_weight_x_1].pipeline(k_grad_weight_x_1.axis[1])
      s[k_grad_weight_x_2].pipeline(k_grad_weight_x_2.axis[1])

    # print(hcl.lower(s))
    return hcl.build(s, target)

hcl_output = hcl.asarray(np.zeros((463,1024,2)), dtype)    
hcl_grad_x = hcl.asarray(np.zeros((463,1024,6)), dtype)    
imgs = [hcl.asarray(_) for _ in imgs]

f = optical_flow(target)
f(*imgs, hcl_output)
print(hcl_output.asnumpy())
print(hcl_grad_x.asnumpy())
