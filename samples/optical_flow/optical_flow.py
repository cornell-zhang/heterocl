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

       sum = hcl.reducer(0, lambda x, y: x + y)

       @hcl.def_([size, (5,), size, size])
       def calc_xy_gradient(input_image, g_w, out_x, out_y):
           rx = hcl.reduce_axis(0, 5, name="rdx")
           ry = hcl.reduce_axis(0, 5, name="rdy")
           out_x = hcl.compute(size, 
               lambda y, x : sum(
                   input_image[y-2,x-rx+2] * g_w[4-rx] / 12, 
                   where=hcl.and_(y>=2, y<height-2, x>=2, x<width-2),
                   axis=rx),
               name="out_x")
           out_y = hcl.compute(size, 
               lambda y, x : sum(
                   input_image[y-ry+2,x-2] * g_w[4-ry] / 12, 
                   where=hcl.and_(y>=2, y<height-2, x>=2, x<width-2),
                   axis=ry),
               name="out_y")

       @hcl.def_([size, size, size, size, size, (5,), size])
       def calc_z_gradient(img0, img1, img2, img3, img4, g_w, grad_z):
           rd_t = hcl.reduce_axis(0, 5, name="time_rdx")
           grad_z = hcl.compute(size, 
               lambda y, x: sum(hcl.select(rd_t==0, img0[y,x],
                                hcl.select(rd_t==1, img1[y,x],
                                hcl.select(rd_t==2, img2[y,x],
                                hcl.select(rd_t==3, img3[y,x], img4[y,x]))))
                                * g_w[rd_t], axis=rd_t), name="grad_z")

       @hcl.def_([size, size, size, (7,), (436,1024,3)])
       def grad_weight_y(grad_x, grad_y, grad_z, g_f, output):
           rd1 = hcl.reduce_axis(0, 7, name="rdx1")
           rd2 = hcl.reduce_axis(0, 7, name="rdx2")
           rd3 = hcl.reduce_axis(0, 7, name="rdx3")
           output = hcl.compute((436,1024,3), 
               lambda y, x, c: 
                   hcl.select(c==0, sum(grad_x[y-rd1+3, x] * g_f[rd1], axis=rd1,
                       where=hcl.and_(y>=3, y<=height-3)), 
                   hcl.select(c==1, sum(grad_y[y-rd2+3, x] * g_f[rd2], axis=rd2,
                       where=hcl.and_(y>=3, y<=height-3)), 
                   sum(grad_x[y-rd3+3, x] * g_f[rd3], axis=rd3,
                       where=hcl.and_(y>=3, y<=height-3)))), name="output")

       @hcl.def_([(436,1024,3), (5,), (436,1024,3)])
       def grad_weight_x(y_filt, g_w, filt_grad):
           rd1 = hcl.reduce_axis(0, 7, name="rdx1")
           rd2 = hcl.reduce_axis(0, 7, name="rdx2")
           rd3 = hcl.reduce_axis(0, 7, name="rdx3")
           filt_grad = hcl.compute((436,1024,3), 
               lambda y, x, c: 
                   hcl.select(c==0, sum(y_filt[y, x-rd1+3,0] * g_w[rd1], axis=rd1,
                       where=hcl.and_(x>=3, x<width-3)), 
                   hcl.select(c==1, sum(y_filt[y, x-rd2+3,1] * g_w[rd2], axis=rd2,
                       where=hcl.and_(x>=3, x<width-3)), 
                   sum(y_filt[y, x-rd3+3,2] * g_w[rd3], axis=rd3,
                       where=hcl.and_(x>=3, x<=width-3)))), name="filt_grad")

       @hcl.def_([(436,1024,3), (436,1024,6)])
       def outer_product(filt_grad, outer):
           outer = hcl.compute((436,1024,6), 
               lambda y, x, c: 
                   hcl.select(c==0, filt_grad[y,x,0] * filt_grad[y,x,0],
                   hcl.select(c==1, filt_grad[y,x,1] * filt_grad[y,x,1],
                   hcl.select(c==2, filt_grad[y,x,2] * filt_grad[y,x,2],
                   hcl.select(c==3, filt_grad[y,x,0] * filt_grad[y,x,1],
                   hcl.select(c==4, filt_grad[y,x,0] * filt_grad[y,x,2], 
                       filt_grad[y,x,1] * filt_grad[y,x,2]))))), name="outer")

       @hcl.def_([(436,1024,6), (3,), (436,1024,6)])
       def tensor_weight_y(outer, t_w, tensor_y):
           rd = hcl.reduce_axis(0, 3, name="rdx_y")
           tensor_y = hcl.compute((436,1024,6), 
               lambda y, x, c: sum(hcl.select(c==0, outer[y-rd+1,x,0],
                                   hcl.select(c==1, outer[y-rd+1,x,1],
                                   hcl.select(c==2, outer[y-rd+1,x,2],
                                   hcl.select(c==3, outer[y-rd+1,x,3], 
                                   hcl.select(c==4, outer[y-rd+1,x,4], 
                                       outer[y-rd,x,5])))))
                                   * t_w[rd], axis=rd, 
                                   where=hcl.and_(y>=1,y<=height-1)), name="tensor_y")

       @hcl.def_([(436,1024,6), (3,), (436,1024,6)])
       def tensor_weight_x(tensor_y, t_w, tensor):
           rd = hcl.reduce_axis(0, 3, name="rdx_x")
           tensor = hcl.compute((436,1024,6), 
               lambda y, x, c: sum(hcl.select(c==0, tensor_y[y,x-rd+1,0],
                                   hcl.select(c==1, tensor_y[y,x-rd+1,1],
                                   hcl.select(c==2, tensor_y[y,x-rd+1,2],
                                   hcl.select(c==3, tensor_y[y,x-rd+1,3], 
                                   hcl.select(c==4, tensor_y[y,x-rd+1,4], 
                                       tensor_y[y,x-rd+1,5])))))
                                   * t_w[rd], axis=rd, 
                                   where=hcl.and_(x>=1,x<=width-1)), name="tensor")

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
