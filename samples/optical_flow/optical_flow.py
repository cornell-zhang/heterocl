import heterocl as hcl
import hlib
import numpy as np
import os, sys
from PIL import Image

# height x width
size = (436, 1024)
height, width = size
hcl.init(hcl.UInt(32))
dtype = hcl.UInt(32)

# setup target using vivado 
tool = hcl.tool.vivado_hls("csim")
target = hcl.platform.zc706

# load ppm image amd convert to grayscale
img0 = Image.open("datasets/current/frame1.ppm").convert("L")
img1 = Image.open("datasets/current/frame2.ppm").convert("L") 
img2 = Image.open("datasets/current/frame3.ppm").convert("L")
img3 = Image.open("datasets/current/frame4.ppm").convert("L")
img4 = Image.open("datasets/current/frame5.ppm").convert("L")

img0 = np.asarray(img0.getdata(), dtype=np.uint8).reshape(img0.size[1], img0.size[0]) 
img1 = np.asarray(img1.getdata(), dtype=np.uint8).reshape(img1.size[1], img1.size[0]) 
img2 = np.asarray(img2.getdata(), dtype=np.uint8).reshape(img2.size[1], img2.size[0]) 
img3 = np.asarray(img3.getdata(), dtype=np.uint8).reshape(img3.size[1], img3.size[0]) 
img4 = np.asarray(img4.getdata(), dtype=np.uint8).reshape(img4.size[1], img4.size[0]) 
imgs = [img0, img1, img2, img3, img4]


def optical_flow(target=target):

    image0 = hcl.placeholder((436,1024), "input_image0")
    image1 = hcl.placeholder((436,1024), "input_image1")
    image2 = hcl.placeholder((436,1024), "input_image2")
    image3 = hcl.placeholder((436,1024), "input_image3")
    image4 = hcl.placeholder((436,1024), "input_image4")
    g_w = hcl.placeholder((5,), "grad_weight")
    g_f = hcl.placeholder((7,), "grad_filter")
    t_f = hcl.placeholder((3,), "tensor_filter")
    output = hcl.placeholder((436,1024,2), "output_image")

    def kernel(img0, img1, img2, img3, img4, g_w, g_f, t_f, output):

       @hcl.def_([size, (5,), size, size])
       def calc_xy_gradient(input_image, g_w, out_x, out_y):
           with hcl.for_(0, height+2, name="r") as r:
             with hcl.for_(0, width+2, name="c") as c:
               sx = hcl.scalar(0, "acc.x")
               sy = hcl.scalar(0, "acc.y")
               with hcl.if_(hcl.and_(r>=4, r<height, c>=4, c<width)):
                 with hcl.for_(0, 5, name="rdx") as i:
                   sx.v += input_image[r-2][c-i] * g_w[4-i]
                   sy.v += input_image[r-i][c-2] * g_w[4-i]
                 out_x[r-2, c-2] = sx.v / 12
                 out_y[r-2, c-2] = sy.v / 12
               with hcl.elif_(hcl.and_(r>=2, c>=2)):
                 out_x[r-2, c-2] = 0
                 out_y[r-2, c-2] = 0

       @hcl.def_([size, size, size, size, size, (5,), size])
       def calc_z_gradient(img0, img1, img2, img3, img4, g_w, grad_z):
           with hcl.for_(0, height, name="y") as y:
             with hcl.for_(0, width, name="x") as x:
               s = hcl.scalar(0, "acc")
               s.v += img0[y, x] * g_w[0]
               s.v += img1[y, x] * g_w[1]
               s.v += img2[y, x] * g_w[2]
               s.v += img3[y, x] * g_w[3]
               s.v += img4[y, x] * g_w[4]
               grad_z[y, x] = s.v / 12

       @hcl.def_([size, size, size, (7,), (436,1024,3)])
       def grad_weight_y(grad_x, grad_y, grad_z, g_f, output):
           with hcl.for_(0, height+3, name="y") as r:
             with hcl.for_(0, width, name="x") as c:
               sx = hcl.scalar(0, "acc.x")
               sy = hcl.scalar(0, "acc.y")
               sz = hcl.scalar(0, "acc.z")
               with hcl.if_(hcl.and_(r>=6, r<=height)):
                 with hcl.for_(0, 7, name="rdx") as i:
                   sx.v += grad_x[r-i][c] * g_f[i]
                   sy.v += grad_y[r-i][c] * g_f[i]
                   sz.v += grad_z[r-i][c] * g_f[i]
                 output[r-3, c, 0] = sx.v
                 output[r-3, c, 1] = sy.v
                 output[r-3, c, 2] = sz.v
               with hcl.elif_(r>3):
                 output[r-3, c, 0] = sx.v
                 output[r-3, c, 1] = sy.v
                 output[r-3, c, 2] = sz.v

       @hcl.def_([(436,1024,3), (5,), (436,1024,3)])
       def grad_weight_x(y_filt, g_w, filt_grad):
           with hcl.for_(0, height, name="r") as r:
             with hcl.for_(0, width+3, name="c") as c:
               sx = hcl.scalar(0, "acc.x")
               sy = hcl.scalar(0, "acc.y")
               sz = hcl.scalar(0, "acc.z")
               with hcl.if_(hcl.and_(c>=6, c<=width)):
                 with hcl.for_(0, 7, name="rdx") as i:
                   sx.v += y_filt[r,c-i,0] * g_w[i]
                   sy.v += y_filt[r,c-i,1] * g_w[i]
                   sz.v += y_filt[r,c-i,2] * g_w[i]
                 filt_grad[r,c-3,0] = sx.v
                 filt_grad[r,c-3,1] = sy.v
                 filt_grad[r,c-3,2] = sz.v
               with hcl.elif_(c>3):
                 filt_grad[r,c-3,0] = sx.v
                 filt_grad[r,c-3,1] = sy.v
                 filt_grad[r,c-3,2] = sz.v

       @hcl.def_([(436,1024,3), (436,1024,6)])
       def outer_product(filt_grad, outer):
           with hcl.for_(0, height, name="r") as r:
             with hcl.for_(0, width+3, name="c") as c:
               outer[r,c,0] = filt_grad[r,c,0] * filt_grad[r,c,0] 
               outer[r,c,1] = filt_grad[r,c,1] * filt_grad[r,c,1] 
               outer[r,c,2] = filt_grad[r,c,2] * filt_grad[r,c,2] 
               outer[r,c,3] = filt_grad[r,c,0] * filt_grad[r,c,1] 
               outer[r,c,4] = filt_grad[r,c,0] * filt_grad[r,c,2] 
               outer[r,c,5] = filt_grad[r,c,1] * filt_grad[r,c,2] 

       @hcl.def_([(436,1024,6), (3,), (436,1024,6)])
       def tensor_weight_y(outer, t_w, tensor_y):
           with hcl.for_(0, height+1, name="r") as r:
             with hcl.for_(0, width, name="c") as c:
               s0 = hcl.scalar(0, "acc.0")
               s1 = hcl.scalar(0, "acc.1")
               s2 = hcl.scalar(0, "acc.2")
               s3 = hcl.scalar(0, "acc.3")
               s4 = hcl.scalar(0, "acc.4")
               s5 = hcl.scalar(0, "acc.5")
               with hcl.if_(hcl.and_(r>=2, c<=height)):
                 with hcl.for_(0, 3, name="i") as i:
                   s0.v += outer[r-i,c,0]* t_w[i]
                   s1.v += outer[r-i,c,1]* t_w[i]
                   s2.v += outer[r-i,c,2]* t_w[i]
                   s3.v += outer[r-i,c,3]* t_w[i]
                   s4.v += outer[r-i,c,4]* t_w[i]
                   s5.v += outer[r-i,c,5]* t_w[i]
               with hcl.if_(r>=1):
                 tensor_y[r-1,c,0] = s0.v 
                 tensor_y[r-1,c,1] = s1.v 
                 tensor_y[r-1,c,2] = s2.v 
                 tensor_y[r-1,c,3] = s3.v 
                 tensor_y[r-1,c,4] = s4.v 
                 tensor_y[r-1,c,5] = s5.v 

       @hcl.def_([(436,1024,6), (3,), (436,1024,6)])
       def tensor_weight_x(tensor_y, t_w, tensor):
           with hcl.for_(0, height, name="r") as r:
             with hcl.for_(0, width+1, name="c") as c:
               s0 = hcl.scalar(0, "acc.0")
               s1 = hcl.scalar(0, "acc.1")
               s2 = hcl.scalar(0, "acc.2")
               s3 = hcl.scalar(0, "acc.3")
               s4 = hcl.scalar(0, "acc.4")
               s5 = hcl.scalar(0, "acc.5")
               with hcl.if_(hcl.and_(c>=2, c<=width)):
                 with hcl.for_(0, 3, name="i") as i:
                   s0.v += tensor_y[r,c-i,0]* t_w[i]
                   s1.v += tensor_y[r,c-i,1]* t_w[i]
                   s2.v += tensor_y[r,c-i,2]* t_w[i]
                   s3.v += tensor_y[r,c-i,3]* t_w[i]
                   s4.v += tensor_y[r,c-i,4]* t_w[i]
                   s5.v += tensor_y[r,c-i,5]* t_w[i]
               with hcl.if_(c>=1):
                 tensor_y[r,c-1,0] = s0.v 
                 tensor_y[r,c-1,1] = s1.v 
                 tensor_y[r,c-1,2] = s2.v 
                 tensor_y[r,c-1,3] = s3.v 
                 tensor_y[r,c-1,4] = s4.v 
                 tensor_y[r-1,c,5] = s5.v 

       @hcl.def_([(436,1024,6), (3,), (436,1024,2)])
       def flow_calc(tensor, t_w, output):
           with hcl.for_(0, height, name="r") as r:
             with hcl.for_(0, width, name="c") as c:
               with hcl.if_(hcl.and_(r>=2, r<height-2, c>=2, c<width-2)):
                 s0 = hcl.scalar(0, "denom")
                 s0.v = tensor[r,c,0]*tensor[r,c,1] - tensor[r,c,3]*tensor[r,c,3]
                 output[r,c,0] = (tensor[r,c,5]*tensor[r,c,3]-tensor[r,c,1]*tensor[r,c,4]) / s0.v
                 output[r,c,1] = (tensor[r,c,4]*tensor[r,c,3]-tensor[r,c,5]*tensor[r,c,0]) / s0.v
               with hcl.else_():
                 output[r,c,0] = 0
                 output[r,c,1] = 0

       grad_x = hcl.compute(size, lambda x, y: 0, name="grad_x")
       grad_y = hcl.compute(size, lambda x, y: 0, name="grad_y")
       grad_z = hcl.compute(size, lambda x, y: 0, name="grad_z")
       y_filt      = hcl.compute((436,1024,3), lambda *args: 0, name="y_filt")
       filt_grad   = hcl.compute((436,1024,3), lambda *args: 0, name="filt")
       out_product = hcl.compute((436,1024,6), lambda *args: 0, name="product")
       tensor_y = hcl.compute((436,1024,6), lambda *args: 0, name="tensor_y")
       tensor   = hcl.compute((436,1024,2), lambda *args: 0, name="tensor")

       calc_xy_gradient(img2, g_w, grad_x, grad_y)
       calc_z_gradient(img0, img1, img2, img3, img4, g_w, grad_z)

       grad_weight_y(grad_x, grad_y, grad_z, g_f, y_filt)
       grad_weight_x(y_filt, g_w, filt_grad)

       outer_product(filt_grad, out_product)
       tensor_weight_y(out_product, t_f, tensor_y)
       tensor_weight_x(tensor_y, t_f, tensor)
       flow_calc(tensor, t_f, output)

    s = hcl.create_schedule([image0, image1, image2, image3, image4, g_w, 
                             g_f, t_f, output], kernel)

    print(hcl.lower(s))
    # print(kernel.test_func.output_x)
    return hcl.build(s, target)

g_w = hcl.asarray(np.array([-1, -8, 0, 8, 1]), dtype)
g_f = hcl.asarray(np.array([0.0755, 0.133, 0.1869, 0.2903, 0.1869, 0.133, 0.0755]), dtype)
t_f = hcl.asarray(np.array([0.3243, 0.3513, 0.3243]), dtype)
hcl_output = hcl.asarray(np.zeros((463,1024,2)), dtype)    
imgs = [hcl.asarray(_) for _ in imgs]

f = optical_flow(target)
f(*imgs, g_w, g_f, t_f, hcl_output)
