import heterocl as hcl
import hlib
import numpy as np
import os, sys
from PIL import Image

# height x width
size = (436, 1024)
height, width = size
hcl.init(hcl.Fixed(32,12))
target = hcl.platform.aws_f1

def optical_flow(target=target):

    image0 = hcl.placeholder((436,1024), "input_image0")
    image1 = hcl.placeholder((436,1024), "input_image1")
    image2 = hcl.placeholder((436,1024), "input_image2")
    image3 = hcl.placeholder((436,1024), "input_image3")
    image4 = hcl.placeholder((436,1024), "input_image4")
    output = hcl.placeholder((436,1024,2), "output_image")

    def kernel(img0, img1, img2, img3, img4, output):

       sum = hcl.reducer(0, lambda x, y: x + y, dtype)

       @hcl.def_([size, size, size])
       def calc_xy_gradient(input_image, grad_x, grad_y):
           g_w = hcl.copy([1, -8, 0, 8, 1], "g_w", hcl.Int())
           rx = hcl.reduce_axis(0, 5, name="rdx")
           ry = hcl.reduce_axis(0, 5, name="rdy")
           def update(y, x):
               with hcl.if_(hcl.and_(y>=2, y<height-2, x>=2, x<width-2)):
                   grad_x[y][x] = sum(input_image[y][x-rx+2] * g_w[rx], axis=rx)
                   grad_y[y][x] = sum(input_image[y-ry+2][x] * g_w[ry], axis=ry)
           hcl.mutate(size, lambda y, x: update(y, x))
           
       @hcl.def_([size, size, size, size, size, size])
       def calc_z_gradient(img0, img1, img2, img3, img4, grad_z):
           g_w = hcl.copy([1, -8, 0, 8, 1], "g_w", hcl.Int())
           hcl.update(grad_z, 
               lambda y, x: (img0[y,x] * g_w[0] +
                             img1[y,x] * g_w[1] +
                             img2[y,x] * g_w[2] +
                             img3[y,x] * g_w[3] +
                             img4[y,x] * g_w[4]) / 12.0)

       # averaging gradients in y dim
       @hcl.def_([size, size, size, (3,436,1024)])
       def grad_weight_y(grad_x, grad_y, grad_z, y_filt):
           g_f = hcl.copy([0.0755, 0.133, 0.1869, 0.2903, \
                           0.1869, 0.133, 0.0755], "g_f", hcl.Float())
           rd = hcl.reduce_axis(0, 7, name="rdx")
           def acc(c, y, x):
               with hcl.if_(hcl.and_(y>=3, y<height-3)):
                   y_filt[c, y, x] = sum(hcl.select(c==0, grad_x[y+rd-3,x],
                               hcl.select(c==1, grad_y[y+rd-3,x],
                               grad_z[y+rd-3,x])) * g_f[rd], axis=rd)
           hcl.mutate(y_filt.shape, lambda c, y, x: acc(c, y, x))

       @hcl.def_([(3,436,1024), (3,436,1024)])
       def grad_weight_x(y_filt, filt_grad):
           g_f = hcl.copy([0.0755, 0.133, 0.1869, 0.2903, \
                           0.1869, 0.133, 0.0755], "g_f", hcl.Float())
           rd = hcl.reduce_axis(0, 7, name="rdx")
           def acc(c, y, x):
               with hcl.if_(hcl.and_(x>=3, x<width-3)):
                   filt_grad[c, y, x] = sum(y_filt[c, y, x+rd-3] * g_f[rd], axis=rd)
           hcl.mutate(filt_grad.shape, lambda c, y, x: acc(c, y, x))

       @hcl.def_([(3,436,1024), (6,436,1024)])
       def outer_product(filt_grad, outer):
           hcl.update(outer, 
               lambda c, y, x: 
                   hcl.select(c==0, filt_grad[0,y,x] * filt_grad[0,y,x],
                   hcl.select(c==1, filt_grad[1,y,x] * filt_grad[1,y,x],
                   hcl.select(c==2, filt_grad[2,y,x] * filt_grad[2,y,x],
                   hcl.select(c==3, filt_grad[0,y,x] * filt_grad[1,y,x],
                   hcl.select(c==4, filt_grad[0,y,x] * filt_grad[2,y,x], 
                                    filt_grad[1,y,x] * filt_grad[2,y,x]))))))

       @hcl.def_([(6,436,1024), (6,436,1024)])
       def tensor_weight_x(tensor_y, tensor):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx_x")
           def acc(c,y, x):
               with hcl.if_(hcl.and_(x>=1, x<width-1)):
                   tensor[c, y, x] = sum(tensor_y[c,y,x+rd-1] * t_w[rd], axis=rd)
           hcl.mutate(tensor.shape, lambda c, y, x: acc(c, y, x))

       @hcl.def_([(6,436,1024), (6,436,1024)])
       def tensor_weight_y(outer, tensor_y):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx_y")
           def acc(c, y, x):
               with hcl.if_(hcl.and_(y>=1, y<height-1)):
                   tensor_y[c, y, x] = hcl.select(sum(outer[c,y+rd-1,x] * t_w[rd], axis=rd)
           hcl.update(tensor_y, lambda c, y, x: )


       @hcl.def_([(6,436,1024), (436,1024,2)])
       def flow_calc(tensor, output):
           with hcl.for_(0, height, name="r") as r:
             with hcl.for_(0, width, name="c") as c:
               with hcl.if_(hcl.and_(r>=2, r<height-2, c>=2, c<width-2)):
                 s0 = hcl.scalar(0, "denom")
                 s0.v = tensor[0,r,c]*tensor[1,r,c] - tensor[3,r,c]*tensor[3,r,c]
                 output[r,c,0] = (tensor[5,r,c]*tensor[3,r,c]-tensor[1,r,c]*tensor[4,r,c]) / s0.v
                 output[r,c,1] = (tensor[4,r,c]*tensor[3,r,c]-tensor[5,r,c]*tensor[0,r,c]) / s0.v

       calc_xy_gradient(image2, grad_x, grad_y)
       calc_z_gradient(image0, image1, image2, image3, image4, grad_z)

       grad_weight_y(grad_x, grad_y, grad_z, y_filt)
       grad_weight_x(y_filt, filt_grad)

       outer_product(filt_grad, out_product)
       tensor_weight_y(out_product, tensor_y)
       tensor_weight_x(tensor_y, tensor)
       flow_calc(tensor, output)

    s = hcl.create_schedule([image0, image1, image2, image2_0, image3, image4, output], kernel)

    if target != "llvm":

      s.to([image4, image3, image2, image2_0, image1, image0], target.xcel)
      s.to(output, target.host)

      s.reuse_at(k_grad_x.y_filt, s[k_grad_x], k_grad_x.axis[2])
      s.reuse_at(k_grad_y.grad_z, s[k_grad_y], k_grad_y.axis[1])
      s.reuse_at(k_tensor_x.tensor_y, s[k_tensor_x], k_tensor_x.axis[2])
      s.reuse_at(k_tensor_y.outer, s[k_tensor_y], k_tensor_y.axis[1])

      s.to([kernel.grad_x, kernel.grad_y], s[k_grad_y], s[k_grad_xy], hcl.Stream.FIFO)
      s.to(kernel.grad_z, s[k_grad_y], s[k_grad_z], hcl.Stream.FIFO)
      s.to(kernel.y_filt, s[k_grad_x], s[k_grad_y], hcl.Stream.FIFO)
      s.to(kernel.filt_grad, s[k_outer], s[k_grad_x], hcl.Stream.FIFO)
      s.to(kernel.outer, s[k_tensor_y], s[k_outer], hcl.Stream.FIFO)
      s.to(kernel.tensor_y, s[k_tensor_x], s[k_tensor_y], hcl.Stream.FIFO)
      s.to(kernel.tensor, s[k_calc_flow], s[k_tensor_x], hcl.Stream.FIFO)

    return hcl.build(s, target)

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
imgs = [img0, img1, img2, img2, img3, img4]

hcl_output = hcl.asarray(np.zeros((463,1024,2)), dtype)    
hcl_grad_x = hcl.asarray(np.zeros((463,1024,6)), dtype)    
imgs = [hcl.asarray(_) for _ in imgs]

f = optical_flow(target)
f(*imgs, hcl_output)
print(hcl_output.asnumpy())
print(hcl_grad_x.asnumpy())
