import numpy as np
import heterocl as hcl
import imageio
#import cv2

#hcl.init()
hcl.init(init_dtype=hcl.Float())
#img = cv2.imread('harry.jpg', cv2.IMREAD_GRAYSCALE)
img = imageio.imread('./images/rose-grayscale.jpg', as_gray=True)

[hmax , vmax] = img.shape

H = 0
V = 0
Vmin = 0 
Hmin = 0 

in_block = hcl.placeholder((hmax,vmax), "input")
out_block = hcl.placeholder((vmax*hmax,), "output")

def zigzag(in_block, out_block):
  h = hcl.scalar(H, "h")
  v = hcl.scalar(V, "v")
  vmin = hcl.scalar(Vmin, "vmin")
  hmin = hcl.scalar(Hmin, "hmin")
  i = 0 
  with hcl.while_(hcl.and_(v < vmax, h < hmax)):
    # going up
    with hcl.if_(((h + v) % 2) == 0):                                
      with hcl.if_(v == vmin):
        out_block[i] = in_block[v, h]                               # if we got to the first line
        with hcl.if_(h == hmax):
          v = v + 1
        with hcl.else_():
          h = h + 1                                        
        i = i + 1

      with hcl.else_():
         # if we got to the last column
        with hcl.if_(hcl.and_(h == hmax -1, v < vmax)):          
          out_block[i] = in_block[v, h] 
          v = v + 1
          i = i + 1
        with hcl.if_(hcl.and_(v > vmin, h < hmax -1)):
          out_block[i] = in_block[v, h] 
          v = v - 1
          h = h + 1
          i = i + 1
    with hcl.else_():                                             # going down
      with hcl.if_(hcl.and_(v == vmax -1, h <= hmax -1)):       # if we got to the last line
        out_block[i] = in_block[v,h]
        h = h + 1
        i = i + 1
      with hcl.else_():                                         # if we got to the first column
        with hcl.if_(h == hmin):
          out_block[i] = in_block[v, h] 
        with hcl.if_(v == vmax -1):
          h = h + 1
        with hcl.else_():
          v = v + 1
        i = i + 1
                                                                      # if we got to the first column
        with hcl.if_(hcl.and_(v < vmax -1, h > hmin)):        # all other cases
          out_block[i] = in_block[v, h] 
          v = v + 1
          h = h - 1
          i = i + 1
    with hcl.if_(hcl.and_(v == vmax-1, h == hmax-1)):             # bottom right element
      out_block[i] = in_block[v, h] 

s = hcl.create_schedule([in_block, out_block], zigzag)
f = hcl.build(s)  

hcl_img = hcl.asarray(img)
#print(hcl_img.shape)
hcl_O = hcl.asarray(np.zeros((vmax*hmax,)))
#print(hcl_O.shape)

f(hcl_img, hcl_O)
