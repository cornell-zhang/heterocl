import heterocl as hcl
import numpy as np
import os, sys
from PIL import Image

# height x width
size = (436, 1024)
height, width = size

hcl.init(hcl.Float())
target = "llvm"
dtype = hcl.Float()

grad_weights  = [1, -8, 0, 8, -1]
grad_filter   = [0.0755, 0.133, 0.1869, 0.2903, 0.1869, 0.133, 0.0755]
tensor_filter = [0.3243, 0.3513, 0.3243]

def optical_flow(target=target):

    image0 = hcl.placeholder(size, "input_image0")
    image1 = hcl.placeholder(size, "input_image1")
    image2 = hcl.placeholder(size, "input_image2")
    image3 = hcl.placeholder(size, "input_image3")
    image4 = hcl.placeholder(size, "input_image4")
    output = hcl.placeholder((436,1024,2), "output_image")

    def kernel(img0, img1, img2, img3, img4, output):

       sum = hcl.reducer(0, lambda x, y: x + y, dtype)

       @hcl.def_([size, size, size])
       def calc_xy_gradient(input_image, grad_x, grad_y):
           g_w = hcl.copy(grad_weights, "g_w", hcl.Int())
           rx = hcl.reduce_axis(0, 5, name="rdx")
           ry = hcl.reduce_axis(0, 5, name="rdy")
           # 1d conv slong x and y axis
           def update(y, x):
               with hcl.if_(hcl.and_(y>=2, y<height-2, x>=2, x<width-2)):
                   grad_x[y,x] = sum(input_image[y,x+rx-2] * g_w[rx], axis=rx)
                   grad_y[y,x] = sum(input_image[y+ry-2,x] * g_w[ry], axis=ry)
           hcl.mutate(size, lambda y, x: update(y, x))
           
       @hcl.def_([size, size, size, size, size, size])
       def calc_z_gradient(img0, img1, img2, img3, img4, grad_z):
           g_w = hcl.copy(grad_weights, "g_w", hcl.Int())
           hcl.update(grad_z, 
               lambda y, x: (img0[y,x] * g_w[0] +
                             img1[y,x] * g_w[1] +
                             img2[y,x] * g_w[2] +
                             img3[y,x] * g_w[3] +
                             img4[y,x] * g_w[4]) / 12.0)

       # averaging gradients in y dim
       @hcl.def_([size, size, size, (436,1024,3)])
       def grad_weight_y(grad_x, grad_y, grad_z, y_filt):
           g_f = hcl.copy(grad_filter, "g_f", hcl.Float())
           # rd1 = hcl.reduce_axis(0, 7, name="rdx1")
           # rd2 = hcl.reduce_axis(0, 7, name="rdx2")
           # rd3 = hcl.reduce_axis(0, 7, name="rdx3")
           def acc(y, x):
               with hcl.if_(hcl.and_(y>=3, y<height-3)):
                   v1 = hcl.scalar(0, "v1")
                   v2 = hcl.scalar(0, "v2")
                   v3 = hcl.scalar(0, "v3")
                   with hcl.for_(0, 7, name="rd") as rd:
                       v1.v += grad_x[y+rd-3,x] * g_f[rd]
                       v2.v += grad_y[y+rd-3,x] * g_f[rd]
                       v3.v += grad_z[y+rd-3,x] * g_f[rd]
                   y_filt[y,x,0] = v1.v 
                   y_filt[y,x,1] = v2.v 
                   y_filt[y,x,2] = v3.v 
                   # y_filt[c,y,x] = sum(
                   #         hcl.select(c==0, grad_x[y+rd-3,x],
                   #         hcl.select(c==1, grad_y[y+rd-3,x],
                   #         grad_z[y+rd-3,x])) * g_f[rd], axis=rd)
           hcl.mutate(size, acc)

       @hcl.def_([(*size,3), (*size,3)])
       def grad_weight_x(y_filt, filt_grad):
           g_f = hcl.copy(grad_filter, "g_f", hcl.Float())
           # rd = hcl.reduce_axis(0, 7, name="rdx")
           def acc(y, x):
               with hcl.if_(hcl.and_(x>=3, x<width-3)):
                   v1 = hcl.scalar(0, "v1")
                   v2 = hcl.scalar(0, "v2")
                   v3 = hcl.scalar(0, "v3")
                   # reuse the reduction loops
                   with hcl.for_(0, 7, name="rd") as rd:
                       v1.v += y_filt[y,x+rd-3,0] * g_f[rd]
                       v2.v += y_filt[y,x+rd-3,1] * g_f[rd]
                       v3.v += y_filt[y,x+rd-3,2] * g_f[rd]
                   filt_grad[y,x,0] = v1.v 
                   filt_grad[y,x,1] = v2.v 
                   filt_grad[y,x,2] = v3.v 
                   # filt_grad[c, y, x] = sum(y_filt[c, y, x+rd-3] * g_f[rd], axis=rd)
           hcl.mutate(size, acc)

       @hcl.def_([(*size,3), (*size,6)])
       def outer_product(filt_grad, outer):
           # save the condition logic
           def update(y, x):
               outer[y,x,0] = filt_grad[0,y,x] * filt_grad[0,y,x]
               outer[y,x,1] = filt_grad[1,y,x] * filt_grad[1,y,x]
               outer[y,x,2] = filt_grad[0,y,x] * filt_grad[2,y,x]
               outer[y,x,3] = filt_grad[0,y,x] * filt_grad[1,y,x]
               outer[y,x,4] = filt_grad[0,y,x] * filt_grad[2,y,x]
               outer[y,x,5] = filt_grad[1,y,x] * filt_grad[2,y,x]
           hcl.mutate(size, update)

           # hcl.update(outer, 
           #     lambda c, y, x: 
           #         hcl.select(c==0, filt_grad[0,y,x] * filt_grad[0,y,x],
           #         hcl.select(c==1, filt_grad[1,y,x] * filt_grad[1,y,x],
           #         hcl.select(c==2, filt_grad[2,y,x] * filt_grad[2,y,x],
           #         hcl.select(c==3, filt_grad[0,y,x] * filt_grad[1,y,x],
           #         hcl.select(c==4, filt_grad[0,y,x] * filt_grad[2,y,x], 
           #                          filt_grad[1,y,x] * filt_grad[2,y,x]))))))

       @hcl.def_([(*size,6), (*size,6)])
       def tensor_weight_x(tensor_y, tensor):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           # rd = hcl.reduce_axis(0, 3, name="rdx_x")
           def acc(y, x):
               with hcl.if_(hcl.and_(x>=1, x<width-1)):
                   v1 = hcl.scalar(0, "v1")
                   v2 = hcl.scalar(0, "v2")
                   v3 = hcl.scalar(0, "v3")
                   v4 = hcl.scalar(0, "v4")
                   v5 = hcl.scalar(0, "v5")
                   v6 = hcl.scalar(0, "v6")
                   with hcl.for_(0, 3, name="rd") as rd:
                       v1.v += tensor_y[y,x+rd-1,0] * t_w[rd]
                       v2.v += tensor_y[y,x+rd-1,1] * t_w[rd]
                       v3.v += tensor_y[y,x+rd-1,2] * t_w[rd]
                       v4.v += tensor_y[y,x+rd-1,3] * t_w[rd]
                       v5.v += tensor_y[y,x+rd-1,4] * t_w[rd]
                       v6.v += tensor_y[y,x+rd-1,5] * t_w[rd]
                   tensor[y,x,0] = v1.v 
                   tensor[y,x,1] = v2.v 
                   tensor[y,x,2] = v3.v 
                   tensor[y,x,3] = v4.v 
                   tensor[y,x,4] = v5.v 
                   tensor[y,x,5] = v6.v 
                   # tensor[y,x,0] = sum(tensor_y[c,y,x+rd-1] * t_w[rd], axis=rd)
           hcl.mutate(size, acc)

       @hcl.def_([(*size,6), (*size,6)])
       def tensor_weight_y(outer, tensor_y):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           # rd = hcl.reduce_axis(0, 3, name="rdx_y")
           def acc(y, x):
               with hcl.if_(hcl.and_(y>=1, y<height-1)):
                   v1 = hcl.scalar(0, "v1")
                   v2 = hcl.scalar(0, "v2")
                   v3 = hcl.scalar(0, "v3")
                   v4 = hcl.scalar(0, "v4")
                   v5 = hcl.scalar(0, "v5")
                   v6 = hcl.scalar(0, "v6")
                   with hcl.for_(0, 3, name="rd") as rd:
                       v1.v += outer[y+rd-1,x,0] * t_w[rd]
                       v2.v += outer[y+rd-1,x,1] * t_w[rd]
                       v3.v += outer[y+rd-1,x,2] * t_w[rd]
                       v4.v += outer[y+rd-1,x,3] * t_w[rd]
                       v5.v += outer[y+rd-1,x,4] * t_w[rd]
                       v6.v += outer[y+rd-1,x,5] * t_w[rd]
                   tensor_y[y,x,0] = v1.v 
                   tensor_y[y,x,1] = v2.v 
                   tensor_y[y,x,2] = v3.v 
                   tensor_y[y,x,3] = v4.v 
                   tensor_y[y,x,4] = v5.v 
                   tensor_y[y,x,5] = v6.v 
                   # tensor_y[c, y, x] = hcl.select(sum(outer[c,y+rd-1,x] * t_w[rd], axis=rd))
           hcl.mutate(size, acc)

       @hcl.def_([(*size,6), (*size,2)])
       def flow_calc(tensor, output):
           with hcl.for_(0, height, name="r") as r:
             with hcl.for_(0, width, name="c") as c:
               with hcl.if_(hcl.and_(r>=2, r<height-2, c>=2, c<width-2)):
                 s0 = hcl.scalar(0, "denom")
                 s0.v = tensor[r,c,0]*tensor[r,c,1] - tensor[r,c,3]*tensor[r,c,3]
                 output[r,c,0] = (tensor[r,c,5]*tensor[r,c,3]-tensor[r,c,1]*tensor[r,c,4]) / s0.v
                 output[r,c,1] = (tensor[r,c,4]*tensor[r,c,3]-tensor[r,c,5]*tensor[r,c,0]) / s0.v

       init = lambda *args: 0
       grad_x = hcl.compute(size, init, name="grad_x")
       grad_y = hcl.compute(size, init, name="grad_y")
       grad_z = hcl.compute(size, init, name="grad_z")


       y_filt      = hcl.compute((*size,3), init, name="y_filt")
       filt_grad   = hcl.compute((*size,3), init, name="filt_grad")
       out_product = hcl.compute((*size,6), init, name="out_product")
       tensor_y    = hcl.compute((*size,6), init, name="tensor_y")
       tensor      = hcl.compute((*size,6), init, name="tensor")

       calc_xy_gradient(image2, grad_x, grad_y)
       hcl.print(image2)
       hcl.print(grad_x)
       calc_z_gradient(image0, image1, image2, image3, image4, grad_z)

       grad_weight_y(grad_x, grad_y, grad_z, y_filt)
       grad_weight_x(y_filt, filt_grad)

       outer_product(filt_grad, out_product)
       tensor_weight_y(out_product, tensor_y)
       tensor_weight_x(tensor_y, tensor)
       flow_calc(tensor, output)

    s = hcl.create_schedule([image0, image1, image2, image3, image4, output], kernel)

    if target != "llvm":

      s.to([image4, image3, image2, image1, image0], target.xcel)
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

    print(hcl.lower(s))
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
imgs = [img0, img1, img2, img3, img4]

hcl_output = hcl.asarray(np.zeros((463,1024,2)), dtype)    
hcl_grad_x = hcl.asarray(np.zeros((463,1024,6)), dtype)    
imgs = [hcl.asarray(_) for _ in imgs]

f = optical_flow(target)
f(*imgs, hcl_output)
print(hcl_output.asnumpy())
print(hcl_grad_x.asnumpy())
