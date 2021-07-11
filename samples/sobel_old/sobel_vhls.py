import heterocl as hcl
from PIL import Image
import numpy as np
import math
import imageio
from urllib.request import urlopen

img = Image.open(urlopen('https://i.imgur.com/646oDya.jpeg'))
width, height = img.size
hcl.init(init_dtype=hcl.Float())

A = hcl.placeholder((height,width,3), "A")
Gx = hcl.placeholder((3,3),"Gx")
Gy = hcl.placeholder((3,3),"Gy")

def sobel(A,Gx,Gy):   
   B = hcl.compute((height,width), lambda x,y: A[x][y][0]+A[x][y][1]+A[x][y][2], "B") 
   r = hcl.reduce_axis(0,3)
   c = hcl.reduce_axis(0,3)
  # D = hcl.compute((height, width), lambda x,y: hcl.select(hcl.and_(x>0,x<(height-1),y>0,y<(width-1)), hcl.sum(B[x+r,y+c]*Gx[r,c],axis=[r,c]), B[x,y]), "xx")
   D = hcl.compute((height-2, width-2), lambda x,y: hcl.sum(B[x+r, y+c]*Gx[r,c], axis=[r,c], name="sum1"), "xx")

   t = hcl.reduce_axis(0, 3)
   g = hcl.reduce_axis(0, 3)
  # E = hcl.compute((height, width), lambda x,y: hcl.select(hcl.and_(x>0,x<(height-1),y>0,y<(width-1)), hcl.sum(B[x+t,y+g]*Gy[t,g],axis=[t,g]), B[x,y]), "yy")
   E = hcl.compute((height-2, width-2), lambda x,y: hcl.sum(B[x+t, y+g]*Gy[t,g], axis=[t,g]), "yy")

   return  hcl.compute((height-2,width-2), lambda x,y:hcl.sqrt(D[x][y]*D[x][y]+E[x][y]*E[x][y])*0.05891867,"Fimg")

s = hcl.create_schedule([A,Gx,Gy],sobel)
# LBX = s.reuse_at(sobel.B._op, s[sobel.xx], sobel.xx.axis[0], "LBX")
# LBY = s.reuse_at(sobel.B._op, s[sobel.yy], sobel.yy.axis[0], "LBY") 
# WBX = s.reuse_at(LBX, s[sobel.xx], sobel.xx.axis[1], "WBX")
# WBY = s.reuse_at(LBY, s[sobel.yy], sobel.yy.axis[1], "WBY")
# s.partition(LBX)
# s.partition(LBY)
# s.partition(WBX)
# s.partition(WBY)
# s.partition(Gx)
# s.partition(Gy)
s[sobel.xx].pipeline(sobel.xx.axis[1])
s[sobel.yy].pipeline(sobel.yy.axis[1])

# f = hcl.build(s)
# npGx = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
# npGy = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
# hcl_Gx = hcl.asarray(npGx)
# hcl_Gy = hcl.asarray(npGy)
# npA = np.array(img)
# hcl_A = hcl.asarray(npA)
# npF = np.zeros((height-2,width-2))
# hcl_F = hcl.asarray(npF)
# f(hcl_A, hcl_Gx, hcl_Gy, hcl_F)
# npF = hcl_F.asnumpy()
# 
# #define array for image
# newimg = np.zeros((height-2, width-2, 3))
# 
# #assign (length, length, length) to each pixel
# for x in range (0, height-2):
#         for y in range (0, width-2):
#                 for z in range (0, 3):
#                         newimg[x,y,z]=npF[x,y]

target = hcl.Platform.xilinx_zc706 
s.to([A,Gx,Gy], target.xcel) 
s.to(sobel.Fimg, target.host)
target.config(compile= "vivado_hls" , mode= "debug" )
print(hcl.build(s, target))

target.config(compile= "vivado_hls" , mode= "csim|csyn" )

# prepare the input data and output placeholder to store the result 
hcl_A1 = hcl.asarray(np.random.randint( 10 , size=(height, width,3))) 
hcl_Gx1 = hcl.asarray(np.random.randint( 10 , size=(3, 3))) 
hcl_Gy2 = hcl.asarray(np.random.randint( 10 , size=(3, 3))) 
hcl_m3 = hcl.asarray(np.zeros((height-2, width-2)))
f1 = hcl.build(s, target) 
f1(hcl_A1, hcl_Gx1,hcl_Gy2, hcl_m3)
# Return a dictionary storing all the HLS results 
report = f1.report(target)
