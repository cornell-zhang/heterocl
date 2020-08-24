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
import imageio

#hcl.init(init_dtype=hcl.Fixed(15,5))
hcl.init(init_dtype=hcl.Float())
#hcl.init(init_dtype=hcl.Int())

path = './images/rose-grayscale.jpg'
img = imageio.imread(path)

fn, fext = os.path.splitext(path)
height, width, rgb = img.shape

def sobel():

  # IO Placeholders
  imgF = hcl.placeholder((height, width,3), "Image")
  Gx = hcl.placeholder((3,3), "Gx")
  Gy = hcl.placeholder((3,3), "Gy")
 
  def sobel_kernel(imgF, Gx, Gy):
    P = hcl.compute((height+2, width+2, 3), lambda x,y,z: 0, "P")

    def pad(x,y,z):
      P[x+1,y+1,z] = imgF[x,y,z]
    hcl.mutate(imgF.shape, lambda x,y,z: pad(x,y,z), "M")

    A = hcl.compute((height+2, width+2), lambda x,y: P[x][y][0] + P[x][y][1] + P[x][y][2], "A") 

    r = hcl.reduce_axis(0, 3)
    c = hcl.reduce_axis(0, 3)
    resX = hcl.compute((height, width), lambda x,y: hcl.sum(A[x+r, y+c]*Gx[r,c], axis=[r,c], name="sum1"), "X")

    t = hcl.reduce_axis(0, 3)
    g = hcl.reduce_axis(0, 3)
    resY = hcl.compute((height, width), lambda x,y: hcl.sum(A[x+t, y+g]*Gy[t,g], axis=[t,g], name="sum2"), "Y")

    R = hcl.compute((height, width), lambda x,y: hcl.sqrt(resX[x][y]*resX[x][y] + resY[x][y]*resY[x][y]), "R")
  
    #norm = hcl.placeholder((), "Norm")
    #norm = 255 / 4328
    norm = hcl.scalar(255/4328)

    return hcl.compute((height, width), lambda x,y: R[x][y] * norm.v, "F")

  s = hcl.create_schedule([imgF, Gx, Gy], sobel_kernel)

  # Memory Customization
  #sA = sobel_kernel.A
  #sX = sobel_kernel.X
  #sY = sobel_kernel.Y

  #LBX = s.reuse_at(sA._op, s[sX], sX.axis[0], "LBX")
  #LBY = s.reuse_at(sA._op, s[sY], sY.axis[0], "LBY")
  #WBX = s.reuse_at(LBX, s[sX], sX.axis[1], "WBX")
  #WBY = s.reuse_at(LBY, s[sY], sY.axis[1], "WBY")
  #s.partition(LBX, dim=1)
  #s.partition(LBY, dim=1)
  #s.partition(WBX)
  #s.partition(WBY)
  #s.partition(Gx)
  #s.partition(Gy)
  #s[sX].pipeline(sX.axis[1])
  #s[sY].pipeline(sY.axis[1])

  #with hcl.if_(target == None):
  #  print('here')
  #  return hcl.build(s)
  #with hcl.else_():
    # Performance Report
  
  target = hcl.platform.zc706  
  s.to([imgF, Gx, Gy], target.xcel)                       
  s.to(sobel_kernel.F, target.host)
                                                        
  target.config(compile="vivado_hls", mode="csim|csyn")

  hcl_img = hcl.asarray(img)
  hcl_Gx = hcl.asarray(np.array([[-1,-2,-1],[0,0,0],[1,2,1]]))
  hcl_Gy = hcl.asarray(np.array([[-1,0,1],[-2,0,2],[-1,0,1]]))
  hcl_F = hcl.asarray(np.zeros((height, width)))

  f = hcl.build(s,target) 
  #f = hcl.build(s)
  
  f(hcl_img, hcl_Gx, hcl_Gy, hcl_F)
  #print(hcl_F.asnumpy())
  report = f.report()

sobel()

# Save the output
#npF = hcl_F.asnumpy()
#npF -= np.amin(npF)
#npF = npF / np.amax(npF) * 255
#npF = npF.astype(np.uint8)
#print(npF)

#imageio.imsave('out_sobel{}'.format(fext), npF)
