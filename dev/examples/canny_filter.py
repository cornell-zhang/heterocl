#
# "Canny Edge Detection Step by Step in Python - Computer Vision", by Sofiane Sahir
# URL: https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
#

import numpy as np
from PIL import Image
import os
import imageio
#==================================================================================================
# 0. Convert into grayscale image
#==================================================================================================
# Algorithm is based on grayscale pictures.

path = './images/rose-grayscale.jpg'
img = Image.open(path).convert('L')
fn, fext = os.path.splitext(path)

width, height = img.size

npImg = np.asarray(img)

#==================================================================================================
# 1. Noise Reduction
#==================================================================================================
# Apply Gaussian blur to smooth it
# kernel size depends on the expected blurring effect (smaller kernel -> less visible is the blur) 
# Example using 5x5 kernel
def gaussian_kernel(size, sigma=1):
  size = int(size)
  x, y = np.mgrid[-size:size+1, -size:size+1]
  normal = 1 / (2.0 * np.pi * sigma ** 2)
  g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
  return g

G = gaussian_kernel(5)

#==================================================================================================
# 2. Gradient Calculation
#==================================================================================================
# Make use of Sobel kernels
from scipy import ndimage

def sobel_filters(img):
  Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
  Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], np.float32)

  Ix = ndimage.filters.convolve(img, Kx)
  Iy = ndimage.filters.convolve(img, Ky)

  # Magnitude of the gradient
  G = np.hypot(Ix, Iy)
  # Normalization
  G = G / G.max() * 255
  # Slope of a gradient
  theta = np.arctan2(Iy, Ix)

  return (G, theta)

(O, theta) = sobel_filters(npImg)

#==================================================================================================
# 3. Non-maximum Suppression
#==================================================================================================
# Mitigate the thicker edges from sobel

def non_max_suppression(img, D):
  # Create a 0-matrix of the same size as gradient intensity matrix
  M, N = img.shape
  Z = np.zeros((M, N), dtype=np.int32)
  angle = D * 180 / np.pi
  angle[angle < 0] += 180
  
  for i in range(1,M-1):
    for j in range(1,N-1):
      # Identify the edge direction based on the angle value from the angle matrix
      try:
        q = 255
        r = 255

	# angle 0
        if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
          q = img[i, j+1]
          r = img[i, j-1]
	#angle 45
        elif (22.5 <= angle[i,j] < 67.5):
          q = img[i+1, j-1]
          r = img[i-1, j+1]
	#angle 90
        elif (67.5 <= angle[i,j] < 112.5):
          q = img[i+1, j]
          r = img[i-1, j]
	#angle 135
        elif (112.5 <= angle[i,j] < 157.5):
          q = img[i-1, j-1]
          r = img[i+1, j+1]
	
	# Check if the pixel in the same direction has a higher intensity than the pixel that
	# is currently processed
        if (img[i,j] >= q) and (img[i,j] >= r):
          Z[i,j] = img[i,j]
        else:
          Z[i,j] = 0
      
      except IndexError as e:
        pass

  return Z # image processed with non-max suppression algorithm

Z = non_max_suppression(O, theta)

#==================================================================================================
# 4. Double Threshold
#==================================================================================================
# Identify 3 kinds of pixels: Strong, Weak, and Non-relevant

# Strong pixels are pixels that have an intensity so high that we are sure they contribute to the final edge.
# Weak pixels are pixels that have an intensity value that is not enough to be considered as strong ones, but yet not small enough to be considered as non-relevant for the edge detection.
# Other pixels are considered as non-relevant for the edge.

#High threshold is used to identify the strong pixels (intensity higher than the high threshold)
#Low threshold is used to identify the non-relevant pixels (intensity lower than the low threshold)
#All pixels having intensity between both thresholds are flagged as weak and the Hysteresis mechanism (next step) will help us identify the ones that could be considered as strong and the ones that are considered as non-relevant.

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
  highThreshold = img.max() * highThresholdRatio
  lowThreshold = highThreshold * lowThresholdRatio

  M, N = img.shape
  res = np.zeros((M,N), dtype=np.int32)

  weak = np.int32(25)
  strong = np.int32(255)

  strong_i, strong_j = np.where(img >= highThreshold)
  zeros_i, zeros_j = np.where(img < lowThreshold)

  weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

  res[strong_i, strong_j] = strong
  res[weak_i, weak_j] = weak

  return (res, weak, strong)

# result image = with only weak and strong 

(res, weak, strong) = threshold(Z)

#==================================================================================================
# 5. Edge Tracking by Hysteresis
#==================================================================================================
# Transforming weak pixels into strong ones, if and only if at least one of the pixels around the one being processed is a strong one

def hysteresis(img, weak, strong=255):
  M, N = img.shape
  for i in range(1, M-1):
    for j in range(1, N-1):
      if (img[i,j] == weak):
        try:
          if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or (img[i, j+1] == strong) or (img [i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
            img[i,j] = strong
          else:
            img[i,j] = 0
        except IndexError as e:
          pass

  return img

resImg = hysteresis(res, weak)
resImg = resImg.astype(np.uint8)
#imageio.imsave('filter_out.jpg', resImg)
#result.save('{}_testing{}'.format(fn, fext))
img.close()
