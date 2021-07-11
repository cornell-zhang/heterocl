from PIL import Image
import heterocl as hcl
import numpy as np
import math
import imageio
from urllib.request import urlopen

hcl.init(init_dtype=hcl.Float())
img = Image.open(urlopen('http://i.stack.imgur.com/8zINU.gif'))
width, height = img.size

A = hcl.placeholder((height,width), "A", dtype=hcl.Float())
Gx = hcl.placeholder((3,3), "Gx",dtype=hcl.Float())
Gy = hcl.placeholder((3,3), "Gy",dtype=hcl.Float())

def sobel(A, Gx, Gy):

   r = hcl.reduce_axis(0,3)
   c = hcl.reduce_axis(0,3)
   B = hcl.compute((height-2,width-2), 
           lambda x,y: hcl.sum(A[x+r,y+c]*Gx[r,c], axis=[r,c], name="sum1"),
           name="B", dtype=hcl.Float())
   t = hcl.reduce_axis(0,3)
   g = hcl.reduce_axis(0,3)

   C = hcl.compute((height-2,width-2), 
           lambda x,y: hcl.sum(A[x+t,y+g]*Gy[t,g], axis=[t,g], name="sum2"),
           name="C", dtype=hcl.Float())
   return hcl.compute((height-2,width-2), 
              lambda x, y :hcl.sqrt(B[x,y]*B[x,y] + C[x,y]*C[x,y])/4328*255,
              name="output", dtype=hcl.Float())

target = hcl.Platform.xilinx_zc706
target.config(compile="sdsoc", mode="sw_sim")

s = hcl.create_schedule([A,Gx,Gy], sobel)
s.to([Gx, Gy, A], target.xcel)
s.to(sobel.output, target.host)
f = hcl.build(s, target)

npA = np.array(img)
npGx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
npGy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
hcl_A = hcl.asarray(npA)
hcl_Gx = hcl.asarray(npGx)
hcl_Gy = hcl.asarray(npGy)

npF = np.zeros((height-2,width-2))
hcl_F = hcl.asarray(npF)

f(hcl_A, hcl_Gx,hcl_Gy, hcl_F)
npF = hcl_F.asnumpy()

newimg = np.zeros((height-2,width-2,3))
for x in range(0, height-2):
	for y in range(0, width-2):
		for z in range(0,3):
			newimg[x,y,z] = npF[x,y]

newimg = newimg.astype(np.uint8)
# imageio.imsave("pic_sobel.jpg", newimg)
