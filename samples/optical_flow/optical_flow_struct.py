import heterocl as hcl
import numpy as np
import os, sys
from PIL import Image

size = (436, 1024)
height, width = size
hcl.init(hcl.Float())
dtype = hcl.Float()
so = hcl.Struct({"fa": hcl.Fixed(32,13), "fb": hcl.Fixed(32, 13)})
sa = hcl.Struct({"fa": hcl.Fixed(64,56), "fb": hcl.Fixed(64, 56)})
sb = hcl.Struct({"fa": hcl.Fixed(32,13), "fb": hcl.Fixed(32,13), "fc": hcl.Fixed(32,13)})
sc = hcl.Struct({"fa": hcl.Fixed(32,27), "fb": hcl.Fixed(32,27), "fc": hcl.Fixed(32,27),
                 "fd": hcl.Fixed(32,27), "fe": hcl.Fixed(32,27), "ff": hcl.Fixed(32,27)})

# tool = hcl.tool.sdaccel
# target = hcl.platform.aws_f1(tool)
# target.xcel.lang = "vhls"
target = "llvm"

def optical_flow(target=target):

    images = []
    images.append(hcl.placeholder(size, "image_0",   dtype=hcl.Fixed(17,9)))
    images.append(hcl.placeholder(size, "image_1",   dtype=hcl.Fixed(17,9)))
    images.append(hcl.placeholder(size, "image_2_0", dtype=hcl.Fixed(17,9)))
    images.append(hcl.placeholder(size, "image_2_1", dtype=hcl.Fixed(17,9)))
    images.append(hcl.placeholder(size, "image_2_2", dtype=hcl.Fixed(17,9)))
    images.append(hcl.placeholder(size, "image_3",   dtype=hcl.Fixed(17,9)))
    images.append(hcl.placeholder(size, "image_4",   dtype=hcl.Fixed(17,9)))
    output = hcl.placeholder(size, "output_image", dtype=so)

    def kernel(img0, img1, img2_0, img2_1, img2_2, img3, img4, output):

       @hcl.def_([size, size], dtypes=[hcl.Fixed(17,9), hcl.Fixed(32,13)])
       def calc_x_gradient(input_image, grad_x):
           g_w = hcl.copy([1, -8, 0, 8, 1], "g_w", hcl.Int())
           rx = hcl.reduce_axis(0, 5, name="rdx")
           sum = hcl.reducer(0, lambda x, y: x + y, dtype=hcl.Fixed(17,9))
           def update(y, x):
               grad_x[y, x+2] = sum(input_image[y, x+rx] * g_w[rx], axis=rx)
           hcl.mutate((height, width-4), lambda y, x: update(y, x))
           
       @hcl.def_([size, size], dtypes=[hcl.Fixed(17,9), hcl.Fixed(32,13)])
       def calc_y_gradient(input_image, grad_y):
           g_w = hcl.copy([1, -8, 0, 8, 1], "g_w", hcl.Int())
           ry = hcl.reduce_axis(0, 5, name="rdy")
           sum = hcl.reducer(0, lambda x, y: x + y, dtype=hcl.Fixed(17,9))
           def update(y, x):
               grad_y[y+2, x] = sum(input_image[y+ry, x] * g_w[ry], axis=ry)
           hcl.mutate((height-4, width), lambda y, x: update(y, x))

       @hcl.def_([size, size, size, size, size, size], 
                 dtypes=[hcl.Fixed(17,9), hcl.Fixed(17,9), hcl.Fixed(17,9), 
                         hcl.Fixed(17,9), hcl.Fixed(17,9), hcl.Fixed(32,13)])
       def calc_z_gradient(img0, img1, img2, img3, img4, grad_z):
           g_w = hcl.copy([1, -8, 0, 8, 1], "g_w", hcl.Int())
           hcl.update(grad_z, 
               lambda y, x: (img0[y,x] * g_w[0] +
                             img1[y,x] * g_w[1] +
                             img2[y,x] * g_w[2] +
                             img3[y,x] * g_w[3] +
                             img4[y,x] * g_w[4]) / 12.0)

       @hcl.def_([size, size, size, size], 
                 dtypes=[hcl.Fixed(32,13), hcl.Fixed(32,13), hcl.Fixed(32,13), sb])
       def grad_pack(grad_x, grad_y, grad_z, pack):
           def update(y, x):
               t = hcl.scalar(0, name="t", dtype=sb)
               t.v.fa = grad_x[y, x]
               t.v.fb = grad_y[y, x]
               t.v.fc = grad_z[y, x]
               pack[y, x] = t.v 
           hcl.mutate((size), lambda y, x: update(y, x))

       @hcl.def_([size, size], dtypes=[sb, sb])
       def grad_weight_y(pack, y_filt):
           g_f = hcl.copy([0.0755, 0.133, 0.1869, 0.2903, \
                           0.1869, 0.133, 0.0755], "g_f", hcl.Fixed(32,13))
           def update(y, x):
               a = hcl.scalar(0, "a")
               b = hcl.scalar(0, "b")
               c = hcl.scalar(0, "c")
               with hcl.for_(0, 7) as rd:
                  t = hcl.scalar(pack[y+rd, x], name="t", dtype=sb)
                  a.v += t.v.fa * g_f[rd] 
                  b.v += t.v.fb * g_f[rd]
                  c.v += t.v.fc * g_f[rd]
               y_filt[y+3, x].fa = a.v 
               y_filt[y+3, x].fb = b.v
               y_filt[y+3, x].fc = c.v
           hcl.mutate((height-6, width), lambda y, x: update(y, x))

       @hcl.def_([size, size], dtypes=[sb, sb])
       def grad_weight_x(y_filt, filt_grad):
           g_f = hcl.copy([0.0755, 0.133, 0.1869, 0.2903, \
                           0.1869, 0.133, 0.0755], "g_f", hcl.Fixed(32,13))
           def update(y, x):
               a = hcl.scalar(0, "a")
               b = hcl.scalar(0, "b")
               c = hcl.scalar(0, "c")
               with hcl.for_(0, 7) as rd:
                  t = hcl.scalar(y_filt[y, x+rd], name="t", dtype=sb)
                  a.v += t.v.fa * g_f[rd] 
                  b.v += t.v.fb * g_f[rd]
                  c.v += t.v.fc * g_f[rd]
               filt_grad[y, x+3].fa = a.v 
               filt_grad[y, x+3].fb = b.v
               filt_grad[y, x+3].fc = c.v
           hcl.mutate((height, width-6), lambda y, x: update(y, x))

       @hcl.def_([size, size], dtypes=[sb, sc])
       def outer_product(filt_grad, out_product):
           def update(y, x):
               t = hcl.scalar(filt_grad[y, x], name="t", dtype=sb)
               r = hcl.scalar(0, dtype=sc)
               r.v.fa = t.v.fa * t.v.fa
               r.v.fb = t.v.fb * t.v.fb
               r.v.fc = t.v.fc * t.v.fc
               r.v.fd = t.v.fa * t.v.fb
               r.v.fe = t.v.fa * t.v.fc
               r.v.ff = t.v.fb * t.v.fc
               out_product[y, x].fa = r.v 
           hcl.mutate(out_product.shape, lambda y, x: update(y, x))

       @hcl.def_([size, size], dtypes=[sc, sc])
       def tensor_weight_y(out_product, tensor_y):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", dtype=hcl.Fixed(32,13))
           def update(y, x):
               a = hcl.scalar(0, "a")
               b = hcl.scalar(0, "b")
               c = hcl.scalar(0, "c")
               d = hcl.scalar(0, "d")
               e = hcl.scalar(0, "e")
               f = hcl.scalar(0, "f")
               with hcl.for_(0, 3) as rd:
                  t = hcl.scalar(out_product[y+rd, x], name="t", dtype=sc)
                  a.v += t.v.fa * t_w[rd] 
                  b.v += t.v.fb * t_w[rd]
                  c.v += t.v.fc * t_w[rd]
                  d.v += t.v.fd * t_w[rd] 
                  e.v += t.v.fe * t_w[rd]
                  f.v += t.v.ff * t_w[rd]
               r = hcl.scalar(0, dtype=sc)
               r.v.fa = a.v
               r.v.fb = b.v
               r.v.fc = c.v
               r.v.fd = d.v
               r.v.fe = e.v
               r.v.ff = f.v
               tensor_y[y+1, x] = r.v
           hcl.mutate((height-2, width), lambda y, x: update(y, x))

       @hcl.def_([size, size], dtypes=[sc, sc])
       def tensor_weight_x(tensor_y, tensor):
           t_w = hcl.copy([0.3243, 0.3513, 0.3243], "t_w", dtype=hcl.Fixed(32,13))
           def update(y, x):
               a = hcl.scalar(0, "a")
               b = hcl.scalar(0, "b")
               c = hcl.scalar(0, "c")
               d = hcl.scalar(0, "d")
               e = hcl.scalar(0, "e")
               f = hcl.scalar(0, "f")
               with hcl.for_(0, 3) as rd:
                  t = hcl.scalar(tensor_y[y, x+rd], name="t", dtype=sc)
                  a.v += t.v.fa * t_w[rd] 
                  b.v += t.v.fb * t_w[rd]
                  c.v += t.v.fc * t_w[rd]
                  d.v += t.v.fd * t_w[rd] 
                  e.v += t.v.fe * t_w[rd]
                  f.v += t.v.ff * t_w[rd]
               r = hcl.scalar(0, dtype=sc)
               r.v.fa = a.v
               r.v.fb = b.v
               r.v.fc = c.v
               r.v.fd = d.v
               r.v.fe = e.v
               r.v.ff = f.v
               tensor[y, x+1] = r.v
           hcl.mutate((height, width-2), lambda y, x: update(y, x))

       @hcl.def_([size, size], dtypes=[sc, sa])
       def flow_calc(tensor, output):
           with hcl.for_(0, height, name="r") as r:
             with hcl.for_(0, width, name="c") as c:
               with hcl.if_(hcl.and_(r>=2, r<height-2, c>=2, c<width-2)):
                 t = hcl.scalar(tensor[r,c], dtype=sc)
                 s = hcl.scalar(t.v.fa * t.v.fb - t.v.fd * t.v.fd, "denom")
                 r = hcl.scalar(0, dtype=so)
                 r.v.fa = (t.v.fe * t.v.fd - t.v.fb * t.v.fe) / s.v
                 r.v.fb = (t.v.fe * t.v.fd - t.v.fe * t.v.fa) / s.v
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

       calc_x_gradient(img2_0, grad_x)
       calc_y_gradient(img2_1, grad_y)
       calc_z_gradient(img0, img1, img2_2, img3, img4, grad_z)

       grad_pack(grad_x, grad_y, grad_z, pack)
       grad_weight_y(pack, y_filt)
       grad_weight_x(y_filt, filt_grad)

       outer_product(filt_grad, out_product)
       tensor_weight_y(out_product, tensor_y)
       tensor_weight_x(tensor_y, tensor)
       flow_calc(tensor, output)

    s = hcl.create_schedule([*images, output], kernel)

    # s.to([*images], target.xcel)
    # s.to(output, target.host)

    # kgx = kernel.calc_x_gradient
    # kgy = kernel.calc_y_gradient
    # kgz = kernel.calc_z_gradient
    # kpc = kernel.grad_pack
    # kwy = kernel.grad_weight_y
    # kwx = kernel.grad_weight_x

    # kop = kernel.outer_product
    # kty = kernel.tensor_weight_y
    # ktx = kernel.tensor_weight_x
    # kfc = kernel.flow_calc

    # s.reuse_at(kgy.input_image, s[kgy], kgy.axis[0])
    # s.reuse_at(kgx.input_image, s[kgx], kgx.axis[1])

    # s.reuse_at(kwy.pack, s[kwy], kwy.axis[0])
    # s.reuse_at(kwx.y_filt, s[kwx], kwx.axis[1])

    # s.reuse_at(ktx.tensor_y, s[ktx], ktx.axis[1])
    # s.reuse_at(kty.out_product, s[kty], kty.axis[0])

    # s.to(kernel.grad_y, s[kpc], s[kgy])
    # s.to(kernel.grad_x, s[kpc], s[kgx])
    # s.to(kernel.grad_z, s[kpc], s[kgz])
    # s.to(kernel.pack,   s[kwy], s[kpc])

    # s.to(kernel.y_filt, s[kwx], s[kwy])
    # s.to(kernel.filt_grad, s[kop], s[kwx])

    # s.to(kernel.out_product, s[kty], s[kop])
    # s.to(kernel.tensor_y, s[ktx], s[kty])
    # s.to(kernel.tensor, s[kfc], s[ktx])

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
imgs = [img0, img1, img2, img2, img2, img3, img4]

hcl_output = hcl.asarray(np.zeros(size))    
imgs = [hcl.asarray(_) for _ in imgs]

f = optical_flow(target)
f(*imgs, hcl_output)
print(hcl_output.asnumpy())
