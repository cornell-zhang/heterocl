#
# sobel-filter.py
# Author: Enzo Lizama
# URL: https://medium.com/@enzoftware/how-to-build-amazing-images-filters-with-python-median-filter-sobel-filter-%EF%B8%8F-%EF%B8%8F-22aeb8e2f540
#

from PIL import Image
import math
import numpy as np
import imageio

path = './images/rose-grayscale.jpg'
img = Image.open(path)

width, height = img.size

#newimg = Image.new("RGB", (width, height), "white")
newimg = np.zeros((height, width))

orig_img = np.asarray(img)
min_val = np.amin(orig_img)
max_val = np.amax(orig_img)

def sobel(width, height):
  for x in range(1, width-1):  # ignore the edge pixels for simplicity (1 to width-1)
      for y in range(1, height-1): # ignore edge pixels for simplicity (1 to height-1)

	# initialise Gx to 0 and Gy to 0 for every pixel
        Gx = 0
        Gy = 0
	
        # top left pixel
        p = img.getpixel((x-1, y-1))
        r = p[0]
        g = p[1]
        b = p[2]

        # intensity ranges from 0 to 765 (255 * 3)
        intensity = r + g + b

        # accumulate the value into Gx, and Gy
        Gx += -intensity
        Gy += -intensity

	# remaining left column
        p = img.getpixel((x-1, y))
        r = p[0]
        g = p[1]
        b = p[2]

        Gx += -2 * (r + g + b)

        p = img.getpixel((x-1, y+1))
        r = p[0]
        g = p[1]
        b = p[2]

        Gx += -(r + g + b)
        Gy += (r + g + b)

        # middle pixels
        p = img.getpixel((x, y-1))
        r = p[0]
        g = p[1]
        b = p[2]

        Gy += -2 * (r + g + b)

        p = img.getpixel((x, y+1))
        r = p[0]
        g = p[1]
        b = p[2]

        Gy += 2 * (r + g + b)
	
        # right column
        p = img.getpixel((x+1, y-1))
        r = p[0]
        g = p[1]
        b = p[2]

        Gx += (r + g + b)
        Gy += -(r + g + b)

        p = img.getpixel((x+1, y))
        r = p[0]
        g = p[1]
        b = p[2]

        Gx += 2 * (r + g + b)

        p = img.getpixel((x+1, y+1))
        r = p[0]
        g = p[1]
        b = p[2]

        Gx += (r + g + b)
        Gy += (r + g + b)
        
        # calculate the length of the gradient (Pythagorean theorem)
        length = math.sqrt((Gx * Gx) + (Gy * Gy))

        # normalise the length of gradient to the range 0 to 255
        length = length / 4328 * 255
        #length = length - min_val
        #length = length / max_val * 255

        length = int(length)

        # draw the length in the edge image
        #newpixel = img.putpixel((length,length,length))
        #newimg.putpixel((x,y),(length,length,length))
        newimg[y,x] = length

sobel(width, height)
newimg = newimg.astype(np.uint8)
#imageio.imsave('filter_out.jpg', newimg)
img.close()
