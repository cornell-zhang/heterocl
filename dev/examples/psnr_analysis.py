#
# Peak Signal to Noise Ratio Analysis of Sobel and Canny Edge Detection
#
# Last Modified: 05/28/2020
#

#import heterocl as hcl
from PIL import Image
import numpy as np
import math
import imageio

#==================================================================================================
# Gather the output from python files
#==================================================================================================
path = './images/rose-grayscale.jpg'
orig = imageio.imread(path) 

height, width, rgb = orig.shape

from sobel_hcl import npF
sobel_edge = npF

from canny_hcl import finalImg
canny_edge = finalImg

orig = np.asarray(orig)

#==================================================================================================
# Computation
#==================================================================================================
def psnr(orig, edge): 
  s = 0
  for m in range(0,height-1):
    for n in range(0,width-1):
      val = (orig[m][n][0].astype(np.uint16) + orig[m][n][1].astype(np.uint16) + orig[m][n][2].astype(np.uint16)) / 3
      #val = val.astype(np.uint8)
      s += (val - edge[m][n]) ** 2
  mse = s / ( height * width )
  psnr = 10 * math.log10( 255 ** 2 / mse )
  return mse, psnr

sobel_mse, sobel_psnr = psnr(orig, sobel_edge)
canny_mse, canny_psnr = psnr(orig, canny_edge)

#==================================================================================================
# Comparison
#==================================================================================================
print('[Sobel] MSE: %0.3f, PSNR: %0.3f' %(sobel_mse, sobel_psnr))
print('[Canny] MSE: %0.3f, PSNR: %0.3f' %(canny_mse, canny_psnr)) 
#assert(sobel_psnr > canny_psnr)
