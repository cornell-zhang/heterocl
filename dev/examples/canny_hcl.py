'''
canny_hcl.py

HeteroCL implementation of Canny Edge Detection

Adapted from "Canny Edge Detection Step by Step in Python - Computer Vision", by Sofiane Sahir
URL: https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
 
Authors: YoungSeok Na, Xiangyi Zhao, Alga Peng, Mira Kim
Last Modified: 08/09/2020
'''

import heterocl as hcl
import numpy as np
import math
import scipy.special
import os
import time
import imageio

#hcl.init(init_dtype=hcl.Fixed(15,5))
hcl.init(init_dtype=hcl.Float())

# Algorithm is based on grayscale pictures.
path = './images/rose-grayscale.jpg'
img = imageio.imread(path, as_gray=True)

fn, fext = os.path.splitext(path)
height, width = img.shape

# Using 5x5 kernel example
k = 2
N = 2*k
size = int(N+1)

sigma = pow(2, N) / (scipy.special.comb(N, k) * math.sqrt(2 * math.pi))
x,y = np.mgrid[-k:k+1, -k:k+1]
normal = 1 / (2.0 * np.pi * sigma**2)
g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

# Non-max suppression constants
a1 = np.tan(np.pi/8)
a2 = np.tan(3*np.pi/8)
a3 = np.tan(5*np.pi/8)
a4 = np.tan(7*np.pi/8)
a5 = np.tan(np.pi)

# Double Threshold constants
maxVal = 255
highThreshold = maxVal * 0.09 # 22.95
lowThreshold = highThreshold * 0.05 # 1.1475
weak = 25
strong = 255

# IO placeholders
imgF = hcl.placeholder((height, width), "Image")
hcl_img = hcl.asarray(np.asarray(img))

H = hcl.placeholder((size, size), "H") # Gaussian kernel of dimensions (2k+1)*(2k+1)
hcl_H = hcl.asarray(g)

Gx = hcl.placeholder((3,3), "Gx") # Horizontal Sobel kernel
hcl_Gx = hcl.asarray(np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]]))

Gy = hcl.placeholder((3,3), "Gy") # Vertical Sobel kernel
hcl_Gy = hcl.asarray(np.asarray([[-1,-2,-1],[0,0,0],[1,2,1]]))

hcl_F = hcl.asarray(np.zeros((height, width)))

# Canny Kernel
def canny_ed(imgF,H,Gx,Gy):
  #P1 = hcl.compute((height+2, width+2), lambda x,y: 0, "P1")
  #P2 = hcl.compute((height+1, width+1), lambda x,y: 0, "P2")
   
  #def pad(x,y):
  #  P[x+1,y+1] = imgF[x,y]
  #hcl.mutate(imgF.shape, lambda x,y: pad(x,y), "M")
  
  # Noise Reduction & Gradient Calculation
  # Gaussian
  h = hcl.reduce_axis(0, size)
  w = hcl.reduce_axis(0, size)
  
  A = hcl.compute((height, width), 
          lambda x,y: hcl.select(hcl.and_(x > (k-1), x < (height-k), y > (k-1), 
                      y < (width-k)), hcl.sum(imgF[x+h, y+w]*H[h,w], axis=[h,w], name="sum1"),
                      imgF[x,y]), "A")
  
  # Sobel
  r = hcl.reduce_axis(0, 3)
  c = hcl.reduce_axis(0, 3)

  X = hcl.compute((height, width), 
          lambda x,y: hcl.select(hcl.and_(x > (k-1), x < (height-k), y > (k-1), 
	              y < (width-k)), hcl.sum(A[x+r, y+c]*Gx[r,c], axis=[r,c], name="sum2"), 
		      A[x,y]), "X")

  t = hcl.reduce_axis(0, 3)
  g = hcl.reduce_axis(0, 3)
                                                                                                                                                                                         
  Y = hcl.compute((height, width), 
          lambda x,y: hcl.select(hcl.and_(x > (k-1), x < (height-k), y > (k-1), 
	              y < (width-k)), hcl.sum(A[x+t, y+g]*Gy[t,g], axis=[t,g], name="sum3"), 
		      A[x,y]), "Y")  
  
  R = hcl.compute((height,width), 
          lambda x,y: hcl.sqrt(X[x][y]*X[x][y] + Y[x][y]*Y[x][y]), "R")

  # Non-maximum Suppression
  def non_max_kernel(x, y):
    q = 255
    r = 255
    pix = Y[x][y] / X[x][y]
    c1 = hcl.and_(pix >= 0, pix < a1)
    c2 = hcl.and_(pix >= a4, pix <= a5)
    c3 = hcl.and_(pix >= a1, pix < a2)
    c4 = hcl.and_(pix >= a2, pix < a3)
    c5 = hcl.and_(pix >= a3, pix < a4)
    
    with hcl.if_(hcl.or_(c1,c2)):
      q = R[x, y+1] 
      r = R[x, y-1]
                                                                
    with hcl.elif_(c3):
      q = R[x+1, y-1]
      r = R[x-1, y+1]
                                                                
    with hcl.elif_(c4):
      q = R[x+1, y]
      r = R[x-1, y]
                                                                
    with hcl.elif_(c5):
      q = R[x-1, y-1]
      r = R[x+1, y+1]
                                                                
    with hcl.else_():
      pass
     
    with hcl.if_(hcl.or_(R[x][y] < q, R[x][y] < r)):
      R[x][y] = 0
  hcl.mutate(R.shape, lambda x,y: non_max_kernel(x,y), "N")

  # Double Threshold
  def threshold_kernel(x, y):
    with hcl.if_(R[x][y] >= lowThreshold):
      with hcl.if_(R[x][y] >= highThreshold):
        R[x][y] = strong
      with hcl.else_():
        R[x][y] = weak
    with hcl.else_():
      R[x][y] = 0
  hcl.mutate(R.shape, lambda x,y: threshold_kernel(x,y), "O")
  
  # Edge Tracking by Hysteresis
  def hyst_kernel(x, y):
    with hcl.if_(R[x][y] == weak):
      with hcl.if_(hcl.or_(R[x-1][y-1] == strong, R[x-1][y] == strong, 
                           R[x-1][y+1] == strong, R[x][y-1] == strong, 
			   R[x][y+1] == strong, R[x+1][y-1] == strong, 
			   R[x+1][y] == strong, R[x+1][y+1] == strong)):
        R[x][y] = strong
      with hcl.else_():
        R[x][y] = 0
  hcl.mutate(R.shape, lambda x,y: hyst_kernel(x,y), "S")

  return R

s = hcl.create_schedule([imgF,H,Gx,Gy],canny_ed)

#==========
# Performance Report

#target = hcl.platform.zc706
#s.to([imgF,H,Gx,Gy], target.xcel)
#s.to(canny_ed.S.R, target.host)

#target.config(compile="vivado_hls", mode="csyn")
#f = hcl.build(s, target)

#==========
f = hcl.build(s)

f(hcl_img,hcl_H,hcl_Gx,hcl_Gy,hcl_F)

#report = f.report()

# Normalization
finalImg = hcl_F.asnumpy()
finalImg -= np.amin(finalImg)
finalImg = finalImg / np.amax(finalImg) * 255
#finalImg = finalImg.astype(np.uint16)
finalImg = finalImg.astype(np.uint8)

# Save the image
#imageio.imsave('out_canny{}'.format(fext), finalImg)
#imageio.imsave('canny.jpg',finalImg)
