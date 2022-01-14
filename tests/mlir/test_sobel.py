import heterocl as hcl
import numpy as np
import math
import imageio

height = 1280
width = 720

A = hcl.placeholder((height,width), "A", dtype=hcl.Float())
Gx = hcl.placeholder((3,3), "Gx",dtype=hcl.Float())
Gy = hcl.placeholder((3,3), "Gy",dtype=hcl.Float())

def sobel(A, Gx, Gy):

    r = hcl.reduce_axis(0,3,"r")
    c = hcl.reduce_axis(0,3,"c")
    B = hcl.compute((height-2,width-2), 
            lambda x,y: hcl.sum(A[x+r,y+c]*Gx[r,c], axis=[r,c]),
            name="B", dtype=hcl.Float())
    t = hcl.reduce_axis(0,3,"t")
    g = hcl.reduce_axis(0,3,"g")

    C = hcl.compute((height-2,width-2), 
            lambda x,y: hcl.sum(A[x+t,y+g]*Gy[t,g], axis=[t,g]),
            name="C", dtype=hcl.Float())
    return hcl.compute((height-2,width-2), 
                lambda x, y: (B[x,y]*B[x,y] + C[x,y]*C[x,y])/4328*255,
                name="Result",
                dtype=hcl.Float()) # remove sqrt

s = hcl.create_schedule([A,Gx,Gy],sobel)
f = hcl.build(s, target="vhls")
print(f)