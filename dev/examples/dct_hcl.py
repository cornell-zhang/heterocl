'''
dct_hcl.py

HeteroCL implementation of DCT transform

Authors: Jayesh, YoungSeok Na
Last Modified: 08/27/2020
'''

import heterocl as hcl
import numpy as np
#import cv2
import imageio
import math

hcl.init(init_dtype=hcl.Float())

dtype = hcl.Float(32)

img_r = imageio.imread('./images/peru.jpeg', as_gray=True)

[h , w] = img_r.shape

block_size = 8

#padding the image
nbh = math.ceil(h/block_size)
nbw = math.ceil(w/block_size)

H =  block_size * nbh
W =  block_size * nbw

# Placeholder
img = hcl.placeholder((h,w), "img")
hcl_img = hcl.asarray(img_r)

# Quantization Matrix Q_50
Q = hcl.placeholder((8,8), "QMat")
hcl_quant_mat = hcl.asarray([[16,11,10,16,24,40,51,61],
                                 [12,12,14,19,26,58,60,55],
                                 [14,13,16,24,40,57,69,56],
                                 [14,17,22,29,51,87,80,62],
                                 [18,22,37,56,68,109,103,77],
                                 [24,35,55,64,81,104,113,92],
                                 [49,64,78,87,103,121,120,101],
                                 [72,92,95,98,112,100,103,99]], dtype=dtype)
# Cosine table
cos = hcl.placeholder((8,8), "cos")
cos_T = hcl.placeholder((8,8), "cos_T")

C = np.asarray([[0.356,0.356,0.356,0.356,0.356,0.356,0.356,0.356],
                [.4904,.4157,.2778,.0975,-.0975,-.2778,-.4157,-.4904],
                [.4619,.1913,-.1913,-.4619,-.4619,-.1913,.1913,.4619],
                [.4157,-.0975,-.4904,-.2778,.2778,.4904,.0975,-.4157],
                [.3536,-.3536,-.3536,.3556,.3536,-.3536,-.3536,.3556],
                [.2778,-.4904,.0975,.4157,-.4157,-.0975,.4904,-.2778],
                [.1913,-.4619,.4619,-.1913,-.1913,.4619,-.4619,.1913],
                [.0975,-.2778,.4157,-.4904,.4904,-.4157,.2778,-.0975]])

hcl_cos = hcl.asarray(C, dtype=dtype)
hcl_cos_T = hcl.asarray(C.T, dtype=dtype)

# Output placeholder
hcl_img_out = hcl.asarray(np.zeros((H,W)))

def DCT(img, Q, cos, cos_T):
 
  #Shift and pad 
  S = hcl.compute(img.shape, lambda x,y: img[x, y] - 128, "S")
  P = hcl.compute((H,W), lambda x,y: 0, "P")
  def loop(x,y):
    P[x,y] = S[x,y]
  hcl.mutate(S.shape, lambda x,y: loop(x,y), "P")

  BP = hcl.compute((8,8), lambda x,y: 0, "BP") #8x8 block
  Z = hcl.compute((H*W,), lambda x: 0, "Z") #Zigzag
  O = hcl.compute((H,W), lambda x,y: 0, "O")

  # Compute start and end row index of the block
  with hcl.for_(0, nbh, name="i") as i:
    row_ind_1 = i * block_size
        
    # Compute start & end column index of the block
    with hcl.for_(0, nbw, name="j") as j:
      col_ind_1 = j * block_size
            
      hcl.update(BP, lambda x,y: P[row_ind_1+x, col_ind_1+y], "block_place")
            
      r = hcl.reduce_axis(0,8,'r')
      I = hcl.compute((8,8), lambda x,y: hcl.sum(cos[x,r] * BP[r,y], axis=r), "I")
      #I = hcl.compute((8,8), lambda x,y: hcl.sum(cos[r,y] * block_place[x,r], axis=r), "I")
      s = hcl.reduce_axis(0,8,'s')
      R = hcl.compute((8,8), lambda x,y: hcl.sum(I[x,s] * cos_T[s,y], axis=s), "R")
      #R = hcl.compute((8,8), lambda x,y: hcl.sum(I[s,y] * cos_T[x,s], axis=s), "R")
            
      def quantization(x,y):
        q_val = hcl.placeholder((), "quant")
        q_val = 1/Q[x][y]
        R[x][y] *= q_val
      hcl.mutate(R.shape, lambda x,y: quantization(x,y), "Qt")
      
      # TODO                     
      # Zigzag Encoding
      # Run Length Encoding

      def update_out(x,y):
        O[row_ind_1+x, col_ind_1+y] = R[x, y]
      hcl.mutate(R.shape, lambda x,y: update_out(x,y), "U")
  return O

s = hcl.create_schedule([img, Q, cos, cos_T], DCT)

#==================================
# Performance Report

target = hcl.platform.zc706
s.to([img, Q, cos, cos_T], target.xcel)
s.to(DCT.O, target.host)

target.config(compile="vivado_hls", mode="csim|csyn")
f = hcl.build(s, target)

#==================================

#f = hcl.build(s)

f(hcl_img, hcl_quant_mat, hcl_cos, hcl_cos_T, hcl_img_out)

report = f.report()

#print(hcl_img_out.asnumpy())

# Save the output
#imageio.imsave('testing.jpeg', hcl_img_out.asnumpy())
