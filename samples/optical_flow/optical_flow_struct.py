import heterocl as hcl
import numpy as np
import os, sys
from PIL import Image

size = (436, 1024)
height, width = size
hcl.init(hcl.Float())
dtype = hcl.Float()
sa = hcl.Struct({"fa": hcl.Float(), "fb": hcl.Float()})
sb = hcl.Struct({"fa": hcl.Int(32), "fb": hcl.Float(), "fc": hcl.Float()})
sc = hcl.Struct({"fa": hcl.Int(8),  "fb": hcl.Float(), "fc": hcl.Float(),
                 "fd": hcl.Int(8),  "fe": hcl.Float(), "ff": hcl.Float()})

tool = hcl.tool.sdaccel
target = hcl.platform.aws_f1(tool)
target.xcel.lang = "vhls"
# target = hcl.platform.aws_f1(hcl.tool.aocl)

def optical_flow(target=target):

    images = [hcl.placeholder(size, "input_image_" + str(_)) for _ in range(5) ]
    output = hcl.placeholder(size, "output_image", dtype=sa)

    def kernel(img0, img1, img2, img3, img4, output):

       sum = hcl.reducer(0, lambda x, y: x + y, dtype)

       @hcl.def_([size, size])
       def calc_x_gradient(input_image, grad_x):
           g_w = hcl.copy([1, -8, 0, 8, 1], "g_w", hcl.Int())
           rx = hcl.reduce_axis(0, 5, name="rdx")
           def update(y, x):
               grad_x[y, x+2] = sum(input_image[y, x+rx] * g_w[rx], axis=rx)
           hcl.mutate((height, width-4), lambda y, x: update(y, x))
           # hcl.update(grad_x, lambda y, x: 
           #     hcl.select(hcl.and_(x>=2, x<width-2), 
           #         sum(input_image[y, x-rx+2] * g_w[rx], axis=rx), 0))
           
       @hcl.def_([size, size])
       def calc_y_gradient(input_image, grad_y):
           g_w = hcl.copy([1, -8, 0, 8, 1], "g_w", hcl.Int())
           ry = hcl.reduce_axis(0, 5, name="rdy")
           def update(y, x):
               grad_y[y+2, x] = sum(input_image[y+ry, x] * g_w[ry], axis=ry)
           hcl.mutate((height-4, width), lambda y, x: update(y, x))
           # hcl.update(grad_y, lambda y, x: 
           #     hcl.select(hcl.and_(y>=2, y<height-2), 
           #         sum(input_image[y-ry+2, x] * g_w[ry], axis=ry), 0))

       @hcl.def_([size, size, size, size, size, size])
       def calc_z_gradient(img0, img1, img2, img3, img4, grad_z):
           g_w = hcl.copy([1, -8, 0, 8, 1], "g_w", hcl.Int())
           hcl.update(grad_z, 
               lambda y, x: (img0[y,x] * g_w[0] +
                             img1[y,x] * g_w[1] +
                             img2[y,x] * g_w[2] +
                             img3[y,x] * g_w[3] +
                             img4[y,x] * g_w[4]) / 12.0)

       @hcl.def_([size, size, size, size], dtypes=[dtype, dtype, dtype, sb])
       def grad_pack(grad_x, grad_y, grad_z, pack):
           def update(y, x):
               t = hcl.scalar(0, name="t", dtype=sb)
               t.v.fa = grad_x[y, x]
               t.v.fb = grad_y[y, x]
               t.v.fc = grad_z[y, x]
               pack[y, x] = t.v 
           hcl.mutate((size), lambda y, x: update(y, x))

       @hcl.def_([size, size])
       def grad_weight_y(pack, y_filt):
           g_f = hcl.copy([0.0755, 0.133, 0.1869, 0.2903, \
                           0.1869, 0.133, 0.0755], "g_f", hcl.Float())
           rd = hcl.reduce_axis(0, 7, name="rdx")
           def update(y, x):
               y_filt[y+3, x] = sum(pack[y+rd,x] * g_f[rd], axis=rd)
           hcl.mutate((height-6, width), lambda y, x: update(y, x))

       @hcl.def_([size, size])
       def grad_weight_x(y_filt, filt_grad):
           g_f = hcl.copy([0.0755, 0.133, 0.1869, 0.2903, \
                           0.1869, 0.133, 0.0755], "g_f", hcl.Float())
           rd = hcl.reduce_axis(0, 7, name="rd")
           def update(y, x):
               filt_grad[y, x+3] = sum(y_filt[y,x+rd] * g_f[rd], axis=rd)
           hcl.mutate((height, width-6), lambda y, x: update(y, x))
           # def acc(y, x):
           #     filt_grad[y, x] = hcl.select(
           #         hcl.and_(x>=3, x<width-3), 
           #             sum(y_filt[y, x+rd-3] * g_f[rd], axis=rd), 0)
           # hcl.mutate(filt_grad.shape, lambda y, x: acc(y, x))

       @hcl.def_([size, size], dtypes=[sb, sc])
       def outer_product(filt_grad, out_product):
           def update(y, x):
               t = hcl.scalar(filt_grad[y, x], name="t", dtype=sb)
               a = hcl.scalar(t.v.fa, "a")
               b = hcl.scalar(t.v.fb, "b")
               c = hcl.scalar(t.v.fc, "c")
               out_product[y, x].fa = a.v * a.v
               out_product[y, x].fb = b.v * b.v
               out_product[y, x].fc = c.v * c.v
               out_product[y, x].fd = a.v * b.v
               out_product[y, x].fe = a.v * c.v
               out_product[y, x].ff = b.v * c.v
           hcl.mutate(out_product.shape, lambda y, x: update(y, x))

       @hcl.def_([size, size])
       def tensor_weight_y(out_product, tensor_y):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx_y")
           hcl.update(tensor_y, lambda y, x: 
               hcl.select(hcl.and_(y>=1, y<height-1), 
                   sum(out_product[y+rd-1,x] * t_w[rd], axis=rd), 0))

       @hcl.def_([size, size])
       def tensor_weight_x(tensor_y, tensor):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", hcl.Float())
           rd = hcl.reduce_axis(0, 3, name="rdx_x")
           hcl.update(tensor, lambda y, x: 
               hcl.select(hcl.and_(x>=1, y<width-1), 
                   sum(tensor_y[y+rd-1,x] * t_w[rd], axis=rd), 0))

       @hcl.def_([size, size], dtypes=[sc, sa])
       def flow_calc(tensor, output):
           with hcl.for_(0, height, name="r") as r:
             with hcl.for_(0, width, name="c") as c:
               with hcl.if_(hcl.and_(r>=2, r<height-2, c>=2, c<width-2)):
                 t = hcl.scalar(tensor[r,c], dtype=sc)
                 a = hcl.scalar(t.v.fa, "a")
                 b = hcl.scalar(t.v.fb, "b")
                 c = hcl.scalar(t.v.fc, "c")
                 d = hcl.scalar(t.v.fd, "d")
                 e = hcl.scalar(t.v.fe, "e")
                 f = hcl.scalar(t.v.ff, "f")
                 s = hcl.scalar(a.v*b.v-d.v*d.v, "denom")
                 r = hcl.scalar(0, dtype=sa)
                 r.v.fa = (e.v * d.v - b.v * e.v) / s.v
                 r.v.fb = (e.v * d.v - e.v * a.v) / s.v
                 output[r,c] = r.v

       init = lambda *args: 0
       grad_x = hcl.compute(size, init, name="grad_x")
       grad_y = hcl.compute(size, init, name="grad_y")
       grad_z = hcl.compute(size, init, name="grad_z")
       pack = hcl.compute(size, init, name="pack")

       y_filt      = hcl.compute(size, init, name="y_filt", dtype=sb)
       filt_grad   = hcl.compute(size, init, name="filt_grad", dtype=sb)
       out_product = hcl.compute(size, init, name="out_product", dtype=sc)
       tensor_y    = hcl.compute(size, init, name="tensor_y", dtype=sc)
       tensor      = hcl.compute(size, init, name="tensor", dtype=sc)

       calc_x_gradient(img2, grad_x)
       calc_y_gradient(img2, grad_y)
       calc_z_gradient(img0, img1, img2, img3, img4, grad_z)

       grad_pack(grad_x, grad_y, grad_z, pack)
       grad_weight_y(pack, y_filt)
       grad_weight_x(y_filt, filt_grad)

       outer_product(filt_grad, out_product)
       tensor_weight_y(out_product, tensor_y)
       tensor_weight_x(tensor_y, tensor)
       flow_calc(tensor, output)

    s = hcl.create_schedule([*images, output], kernel)

    s.to([*images], target.xcel)
    s.to(output, target.host)

    kgx = kernel.calc_x_gradient
    kgy = kernel.calc_y_gradient
    kgz = kernel.calc_z_gradient
    kpc = kernel.grad_pack
    kwy = kernel.grad_weight_y
    kwx = kernel.grad_weight_x

    kop = kernel.outer_product
    kty = kernel.tensor_weight_y
    ktx = kernel.tensor_weight_x
    kfc = kernel.flow_calc

    s.reuse_at(kgy.input_image, s[kgy], kgy.axis[0])
    s.reuse_at(kgx.input_image, s[kgx], kgx.axis[1])

    s.reuse_at(kwy.pack, s[kwy], kwy.axis[0])
    s.reuse_at(kwx.y_filt, s[kwx], kwx.axis[1])

    s.reuse_at(ktx.tensor_y, s[ktx], ktx.axis[0])
    s.reuse_at(kty.out_product, s[kty], kty.axis[0])

    # s.to(kernel.grad_y, s[kpc], s[kgy])
    # s.to(kernel.grad_x, s[kpc], s[kgx])
    # s.to(kernel.grad_z, s[kpc], s[kgz])
    # s.to(kernel.pack,   s[kwy], s[kpc])

    # s.to(kernel.y_filt, s[kwx], s[kwy])
    # s.to(kernel.filt_grad, s[kop], s[kwx])

    # s.to(kernel.out_product, s[kty], s[kop])
    # s.to(kernel.tensor_y, s[ktx], s[kty])
    s.to(kernel.tensor, s[kfc], s[ktx])

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

hcl_output = hcl.asarray(np.zeros(size), dtype=sa)    
imgs = [hcl.asarray(_) for _ in imgs]

f = optical_flow(target)
f(*imgs, hcl_output)
print(hcl_output.asnumpy())
