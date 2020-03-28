import heterocl as hcl
import numpy as np
import os, sys
from PIL import Image
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

size = (436, 1024)
height, width = size
hcl.init(hcl.Float())
dtype = hcl.Float()
target = "llvm"

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
           g_w = hcl.copy([1, -8, 0, 8, -1], "g_w", hcl.Int())
           rx = hcl.reduce_axis(0, 5, name="rdx")
           ry = hcl.reduce_axis(0, 5, name="rdy")
           def update(y, x):
               with hcl.if_(hcl.and_(y>=2, y<height-2, x>=2, x<width-2)):
                   grad_x[y, x] = sum(input_image[y, x-rx+2] * g_w[rx], axis=rx) / 12
                   grad_y[y, x] = sum(input_image[y-ry+2, x] * g_w[ry], axis=ry) / 12
           hcl.mutate(size, lambda y, x: update(y, x))
           
       @hcl.def_([size, size, size, size, size, size])
       def calc_z_gradient(img0, img1, img2, img3, img4, grad_z):
           g_w = hcl.copy([1, -8, 0, 8, -1], "g_w", hcl.Int())
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
                   y_filt[c,y,x] = sum(hcl.select(c==0, grad_x[y+rd-3,x],
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
       def tensor_weight_y(outer, tensor_y):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdy")
           def acc(c, y, x):
               with hcl.if_(hcl.and_(y>=1, y<height-1)):
                   tensor_y[c, y, x] = sum(outer[c,y+rd-1,x] * t_w[rd], axis=rd)
           hcl.mutate(tensor_y.shape, lambda c, y, x: acc(c, y, x))

       @hcl.def_([(6,436,1024), (6,436,1024)])
       def tensor_weight_x(tensor_y, tensor):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx")
           def acc(c,y, x):
               with hcl.if_(hcl.and_(x>=1, x<width-1)):
                   tensor[c,y,x] = sum(tensor_y[c,y,x+rd-1] * t_w[rd], axis=rd)
           hcl.mutate(tensor.shape, lambda c, y, x: acc(c, y, x))


       @hcl.def_([(6,436,1024), (436,1024,2)])
       def flow_calc(tensor, output):
           with hcl.for_(0, height, name="r") as r:
             with hcl.for_(0, width, name="c") as c:
               with hcl.if_(hcl.and_(r>=2, r<height-2, c>=2, c<width-2)):
                 s0 = hcl.scalar(0, "denom")
                 s0.v = tensor[0,r,c]*tensor[1,r,c] - tensor[3,r,c]*tensor[3,r,c]
                 output[r,c,0] = (tensor[5,r,c]*tensor[3,r,c]-tensor[1,r,c]*tensor[4,r,c]) / s0.v
                 output[r,c,1] = (tensor[4,r,c]*tensor[3,r,c]-tensor[5,r,c]*tensor[0,r,c]) / s0.v

       grad_x = hcl.compute(size, lambda *args: 0, "grad_x")
       grad_y = hcl.compute(size, lambda *args: 0, "grad_y")
       grad_z = hcl.compute(size, lambda *args: 0, "grad_z")

       y_filt = hcl.compute((3,436,1024), lambda *args: 0, "y_filt")
       filt_grad = hcl.compute((3,436,1024), lambda *args: 0, "filt_grad")
       out_product = hcl.compute((6,436,1024), lambda *args: 0, "out_product")

       tensor_y = hcl.compute((6,436,1024), lambda *args: 0, "tensor_y")
       tensor   = hcl.compute((6,436,1024), lambda *args: 0, "tensor")

       calc_xy_gradient(image2, grad_x, grad_y)
       calc_z_gradient(image0, image1, image2, image3, image4, grad_z)

       grad_weight_y(grad_x, grad_y, grad_z, y_filt)
       grad_weight_x(y_filt, filt_grad)

       outer_product(filt_grad, out_product)
       tensor_weight_y(out_product, tensor_y)
       tensor_weight_x(tensor_y, tensor)
       flow_calc(tensor, output)

    s = hcl.create_schedule([image0, image1, image2, image3, image4, output], kernel)
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

hcl_output = hcl.asarray(np.zeros((436,1024,2)), dtype)    
imgs = [hcl.asarray(_) for _ in imgs]

f = optical_flow(target)
f(*imgs, hcl_output)
out = hcl_output.asnumpy()
out_img = Image.fromarray(img0, "L")
out_img.save("test.png")

# checking output result 
f = open("datasets/current/ref.flo", "rb")
magic = np.fromfile(f, np.float32, count=1)
assert magic == 202021.25, "invalid flow format"
w, h = np.fromfile(f, np.int32, count=2)
data = np.fromfile(f, np.float32, count=2*w*h)
ref = np.resize(data, (h, w, 2))

accum_err = 0
for y in range(436):
    for x in range(1024):
        out_x = out[y, x, 0]
        out_y = out[y, x, 1]
        if (out_x * out_x + out_y * out_y > 25):
            out_x = out_y = 1e10
        out_deg = np.arctan2(-1 * out_y, -1 * out_x) * 180 / np.pi 
        ref_x, ref_y = ref[y, x, 0], ref[y, x, 1]
        ref_deg = np.arctan2(-1 * ref_y, -1 * ref_x)
        error = out_deg - ref_deg
        accum_err += abs(error)
 
print("Average error:", accum_err / (436 *1024))

