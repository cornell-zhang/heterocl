'''
sobel_hcl.py

HeteroCL implementation of Sobel Edge Detection Algorithm

Adapted from "How to build amazing image filters with Python - Median filter, Sobel filter", by Enzo Lizama
URL: https://medium.com/@enzoftware/how-to-build-amazing-images-filters-with-python-median-filter-sobel-filter-%EF%B8%8F-%EF%B8%8F-22aeb8e2f540

Authors: YoungSeok Na, Xiangyi Zhao, Alga Peng, Mira Kim
Last Modified: 08/09/2020
'''

import heterocl as hcl
import numpy as np
import math
import os
import time
import imageio

#hcl.init(init_dtype=hcl.Fixed(15,5))
hcl.init(init_dtype=hcl.Float())

path = './images/rose-grayscale.jpg'
img = imageio.imread(path)

fn, fext = os.path.splitext(path)
height, width, rgb = img.shape

# Padding
orig_img = np.asarray(img)
padded_img = np.zeros((height+2, width+2, 3))
for x in range (0, height):
  for y in range (0, width):
    padded_img[x+1, y+1] = orig_img[x, y]

for x in range (0, height):
  padded_img[x+1,0] = orig_img[x,0]
  padded_img[x+1, width+1] = orig_img[x, width-1]

for y in range (0, width):
  padded_img[0,y+1] = orig_img[0,y]
  padded_img[height+1, y+1] = orig_img[height-1, y]

padded_img[0,0] = orig_img[0,0]
padded_img[height+1,0] = orig_img[height-1, 0]
padded_img[0, width+1] = orig_img[0, width-1]
padded_img[height+1, width+1] = orig_img[height-1, width-1]

# IO Placeholders
imgF = hcl.placeholder((height+2, width+2, 3), "Image")
Gx = hcl.placeholder((3,3), "Gx")
Gy = hcl.placeholder((3,3), "Gy")

hcl_img = hcl.asarray(padded_img)
hcl_Gx = hcl.asarray(np.array([[-1,-2,-1],[0,0,0],[1,2,1]]))
hcl_Gy = hcl.asarray(np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
hcl_F = hcl.asarray(np.zeros((height, width)))

# Sobel Kernel
def sobel(imgF, Gx, Gy):
 
  A = hcl.compute((height+2, width+2), lambda x,y: imgF[x][y][0] + imgF[x][y][1] + imgF[x][y][2], "A")
  
  r = hcl.reduce_axis(0, 3)
  c = hcl.reduce_axis(0, 3)
  resX = hcl.compute((height, width), lambda x,y: hcl.sum(A[x+r, y+c]*Gx[r,c], axis=[r,c], name="sum1"), "X")

  t = hcl.reduce_axis(0, 3)
  g = hcl.reduce_axis(0, 3)
  resY = hcl.compute((height, width), lambda x,y: hcl.sum(A[x+t, y+g]*Gy[t,g], axis=[t,g], name="sum2"), "Y")

  R = hcl.compute((height, width), lambda x,y: hcl.sqrt(resX[x][y]*resX[x][y] + resY[x][y]*resY[x][y]), "R")
  
  norm = hcl.placeholder((), "Norm")
  norm = 255 / 4328
  
  return hcl.compute((height, width), lambda x,y: R[x][y] * norm, "F")

s = hcl.create_schedule([imgF, Gx, Gy], sobel)

#==================================
# Memory Customization

#LBX = s.reuse_at(sobel.A, s[sobel.X], sobel.X.axis[0], "LBX")
#LBY = s.reuse_at(sobel.A, s[sobel.Y], sobel.Y.axis[0], "LBY")
#WBX = s.reuse_at(LBX, s[sobel.X], sobel.X.axis[1], "WBX")
#WBY = s.reuse_at(LBY, s[sobel.Y], sobel.Y.axis[1], "WBY")
#s.partition(LBX, dim=1)
#s.partition(LBY, dim=1)
#s.partition(WBX)
#s.partition(WBY)
#s.partition(Gx)
#s.partition(Gy)
#s[sobel.X].pipeline(sobel.X.axis[1])
#s[sobel.Y].pipeline(sobel.Y.axis[1])

#==================================
  
#==================================
# Performance Report

target = hcl.platform.zc706
s.to([imgF, Gx, Gy], target.xcel)
s.to(sobel.F, target.host)

target.config(compile="vivado_hls", mode="csim|csyn")
f = hcl.build(s, target)

#==================================

#f = hcl.build(s)
f(hcl_img, hcl_Gx, hcl_Gy, hcl_F)

report = f.report()

# Save the output
npF = hcl_F.asnumpy()
#npF -= np.amin(npF)
#npF = npF / np.amax(npF) * 255
npF = npF.astype(np.uint16)

#imageio.imsave('out_sobel{}'.format(fext), npF)
#imageio.imsave('hcl_out.jpg', npF)
