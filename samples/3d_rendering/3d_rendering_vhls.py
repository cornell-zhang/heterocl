import heterocl as hcl 
import numpy as np 
import pandas as pd
import time

MAX_X = 256
MAX_Y = 256

input_data = np.array(pd.read_csv('data.csv'))
num_3d_triangles = input_data.shape[0] #number of 3d triangles in the 3d object

#triangle_3d is an array of the 3D triangle: (x0,y0,z0,x1,y1,z1,x2,y2,z2)
#triangle_2d is an array of the 2D triangle: (x0,y0,x1,y1,x2,y2,z)
def projection(triangle_3d,triangle_2d,i):
    #x0: x-y plane&x-z plane:x0; y-z plane:z0
    with hcl.if_(i==0):
        with hcl.if_(angle[0]==2):
            #return triangle_3d[2]
            triangle_2d[i]=triangle_3d[2]
        with hcl.else_():
            #return triangle_3d[0]
            triangle_2d[i]=triangle_3d[0]
    
    #y0: x-y plane&y-z plane:y0; x-z plane:z0
    with hcl.if_(i==1):
        with hcl.if_(angle[0]==1):
            #return triangle_3d[2]
            triangle_2d[i]=triangle_3d[2]
        with hcl.else_():
            #return triangle_3d[1]
            triangle_2d[i]=triangle_3d[1]
    
    #x1: x-y plane&x-z plane:x1; y-z plane:z1
    with hcl.if_(i==2):
        with hcl.if_(angle[0]==2):
            #return triangle_3d[5]
            triangle_2d[i]=triangle_3d[5]
        with hcl.else_():
            #return triangle_3d[3]
            triangle_2d[i]=triangle_3d[3]
    
    #y1: x-y plane&y-z plane:y1; x-z plane:z1
    with hcl.if_(i==3):
        with hcl.if_(angle[0]==1):
            #return triangle_3d[5]
            triangle_2d[i]=triangle_3d[5]
        with hcl.else_():
            #return triangle_3d[4]
            triangle_2d[i]=triangle_3d[4]
            
    #x2: x-y plane&x-z plane:x2; y-z plane:z2
    with hcl.if_(i==4):
        with hcl.if_(angle[0]==2):
            #return triangle_3d[8]
            triangle_2d[i]=triangle_3d[8]
        with hcl.else_():
            #return triangle_3d[6]
            triangle_2d[i]=triangle_3d[6]
            
    #y2: x-y plane&y-z plane:y2; x-z plane:z2
    with hcl.if_(i==5):
        with hcl.if_(angle[0]==1):
            #return triangle_3d[8]
            triangle_2d[i]=triangle_3d[8]
        with hcl.else_():
            #return triangle_3d[7]
            triangle_2d[i]=triangle_3d[7]
    
    #z: x-y plane:(z0+z1+z2)/3; y-z plane:(y0+y1+y2)/3; x-z plane:(x0+x1+x2)/3
    with hcl.if_(i==6):
        with hcl.if_(angle[0]==0):
            #return triangle_3d[2]/3+triangle_3d[5]/3+triangle_3d[8]/3
            triangle_2d[i]=triangle_3d[2]/3+triangle_3d[5]/3+triangle_3d[8]/3
        with hcl.elif_(angle[0]==1):
            #return triangle_3d[1]/3+triangle_3d[4]/3+triangle_3d[7]/3
            triangle_2d[i]=triangle_3d[1]/3+triangle_3d[4]/3+triangle_3d[7]/3
        with hcl.elif_(angle[0]==2):
            #return triangle_3d[0]/3+triangle_3d[3]/3+triangle_3d[6]/3
            triangle_2d[i]=triangle_3d[0]/3+triangle_3d[3]/3+triangle_3d[6]/3
            
            
#fragment is a 500*4 array containing pixels in the 2d triangle: x,y,z,color    
def rasterization(frag_cntr,triangle_2d,fragment):
    x0 = hcl.compute((1,),lambda x:triangle_2d[0],"x0")
    y0 = hcl.compute((1,),lambda x:triangle_2d[1],"y0")
    x1 = hcl.compute((1,),lambda x:triangle_2d[2],"x1")
    y1 = hcl.compute((1,),lambda x:triangle_2d[3],"y1")
    x2 = hcl.compute((1,),lambda x:triangle_2d[4],"x2")
    y2 = hcl.compute((1,),lambda x:triangle_2d[5],"y2")
    z  = hcl.compute((1,),lambda x:triangle_2d[6],"z")
    
    #   Determine whether three vertices of a trianlge 
    #   (x0,y0) (x1,y1) (x2,y2) are in clockwise order by Pineda algorithm
    #   if so, return cw > 0
    #   else if three points are in line, return cw == 0
    #   else in counterclockwise order, return cw < 0
    cw = hcl.compute((1,),lambda x:(x2[0]-x0[0])*(y1[0]-y0[0])-(y2[0]-y0[0])*(x1[0]-x0[0]),"cw")
    #frag_cntr counts the pixels
    with hcl.if_(cw[0] == 0):
        frag_cntr[0] = 0
    with hcl.elif_(cw[0] < 0):
        tmp_x = hcl.scalar(x0[0])
        tmp_y = hcl.scalar(y0[0])
        x0[0] = x1[0]
        y0[0] = y1[0]
        x1[0] = tmp_x.v
        y1[0] = tmp_y.v
    
    #find min_x,max_x,min_y,max_y in the 2d triangle
    min_x = hcl.scalar(0)
    max_x = hcl.scalar(0)
    min_y = hcl.scalar(0)
    max_y = hcl.scalar(0)
    with hcl.if_(x0[0]<x1[0]):
        with hcl.if_(x2[0]<x0[0]):
            min_x.v = x2[0]
        with hcl.else_():
            min_x.v = x0[0]
    with hcl.else_():
        with hcl.if_(x2[0]<x1[0]):
            min_x.v = x2[0]
        with hcl.else_():
            min_x.v = x1[0]
    
    with hcl.if_(x0[0]>x1[0]):
        with hcl.if_(x2[0]>x0[0]):
            max_x.v = x2[0]
        with hcl.else_():
            max_x.v = x0[0]
    with hcl.else_():
        with hcl.if_(x2[0]>x1[0]):
            max_x.v = x2[0]
        with hcl.else_():
            max_x.v = x1[0]
            
    with hcl.if_(y0[0]<y1[0]):
        with hcl.if_(y2[0]<y0[0]):
            min_y.v = y2[0]
        with hcl.else_():
            min_y.v = y0[0]
    with hcl.else_():
        with hcl.if_(y2[0]<y1[0]):
            min_y.v = y2[0]
        with hcl.else_():
            min_y.v = y1[0]
            
    with hcl.if_(y0[0]>y1[0]):
        with hcl.if_(y2[0]>y0[0]):
            max_y.v = y2[0]
        with hcl.else_():
            max_y.v = y0[0]
    with hcl.else_():
        with hcl.if_(y2[0]>y1[0]):
            max_y.v = y2[0]
        with hcl.else_():
            max_y.v = y1[0]
            
    color = hcl.scalar(100,"color")

    # i: size of pixels in the triangle
    i = hcl.scalar(0,dtype=hcl.Int())
    with hcl.Stage("S1"):
        with hcl.for_(min_y,max_y) as y:
            with hcl.for_(min_x,max_x) as x:
                pi0 = hcl.compute((1,),lambda a:(x - x0[0]) * (y1[0] - y0[0]) - (y - y0[0]) * (x1[0] - x0[0]))
                pi1 = hcl.compute((1,),lambda a:(x - x1[0]) * (y2[0] - y1[0]) - (y - y1[0]) * (x2[0] - x1[0]))
                pi2 = hcl.compute((1,),lambda a:(x - x2[0]) * (y0[0] - y2[0]) - (y - y2[0]) * (x0[0] - x2[0]))
                # if pi0, pi1 and pi2 are all non-negative, the pixel is in the triangle
                with hcl.if_(hcl.and_(pi0 >= 0,pi1 >= 0,pi2 >= 0)):
                    fragment[i][0] = x
                    fragment[i][1] = y
                    fragment[i][2] = z[0]
                    fragment[i][3] = color.v
                    i.v += 1
    frag_cntr[0] = i.v
 
# pixels is a 500*3 array containing pixels that need to be updated: x,y,color
def zculling(size_pixels,size,fragment,z_buffer,pixels):
    pixel_cntr = hcl.scalar(0,dtype=hcl.Int())

    with hcl.Stage("S2"):
        with hcl.for_(0,size) as n:
            x = hcl.scalar(fragment[n][0],dtype=hcl.Int())
            y = hcl.scalar(fragment[n][1],dtype=hcl.Int())
            z = hcl.scalar(fragment[n][2])
            color = hcl.scalar(fragment[n][3])
            with hcl.if_( z < z_buffer[y][x] ):
                pixels[pixel_cntr][0] = x.v
                pixels[pixel_cntr][1] = y.v
                pixels[pixel_cntr][2] = color.v
                pixel_cntr.v += 1
                z_buffer[y][x] = z.v
    size_pixels[0] = pixel_cntr.v
    
def coloringFB(i,pixels,frame_buffer):
    x = hcl.scalar(pixels[i][0],dtype=hcl.Int())
    y = hcl.scalar(pixels[i][1],dtype=hcl.Int())
    frame_buffer[x][y] = pixels[i][2]

def rendering(triangle_3ds,angle):
    z_buffer = hcl.compute((MAX_X,MAX_Y),lambda x,y:255,"z_buffer")
    frame_buffer = hcl.compute((MAX_X,MAX_Y), lambda x,y:0, "frame_buffer")

    def loop_body(m):
        triangle_3d = hcl.compute((9,),lambda x:triangle_3ds[m][x],"triangle_3d_")
        fragment = hcl.compute((500,4),lambda x,y:0, "fragment")
        pixels = hcl.compute((500,3),lambda x,y:0, "pixels")
        triangle_2d = hcl.compute((7,),lambda x:0,"triangle_2d")
        frag_cntr = hcl.compute((1,),lambda x:0,"frag_cntr")
        size_pixels = hcl.compute((1,),lambda x:0,"size_pixels")
    
        # 1st Stage Projection
        hcl.mutate((7,),lambda x: projection(triangle_3d,triangle_2d,x),"twod_update")
        
        # 2nd Stage Rasterization:update fragment
        hcl.mutate((1,),lambda x:rasterization(frag_cntr,triangle_2d,fragment))
        
        # 3rd Stage Z-culling:update z_buffer,pixels
        hcl.mutate((1,),lambda x: zculling(size_pixels,frag_cntr[0],fragment,z_buffer,pixels))
        
        # coloring frame buffer
        hcl.mutate((size_pixels[0],), lambda x: coloringFB(x,pixels,frame_buffer))
        
    hcl.mutate((num_3d_triangles,), lambda m: loop_body(m),"main_body")
    
    return frame_buffer

hcl.init(hcl.Int())

triangle_3d = hcl.placeholder((num_3d_triangles,9),"triangle_3d",hcl.Int())
angle = hcl.placeholder((1,),"angle",hcl.Int())
    
s = hcl.create_schedule([triangle_3d,angle],rendering)

main_body = rendering.main_body
twod_update = main_body.twod_update

#s[twod_update].parallel(twod_update.axis[0])
#s[main_body].pipeline(main_body.axis[0])
#print(hcl.build(s, target="vhls"))

target = hcl.platform.zc706
s.to([triangle_3d, angle], target.xcel)
s.to(rendering.main_body.frame_buffer, target.host)
target.config(compile="vivado_hls", mode="csim|csyn")

f = hcl.build(s,target=target)

_triangle_3d = hcl.asarray(np.array(input_data),dtype=hcl.Int())
_angle = hcl.asarray(np.array([0]),dtype=hcl.Int())
_frame_buffer = hcl.asarray(np.zeros([MAX_X,MAX_Y]),dtype=hcl.Int())

start = time.time()
f(_triangle_3d,_angle,_frame_buffer)
end = time.time()
output = _frame_buffer.asnumpy()

for j in range(MAX_X-1,-1,-1):
    for i in range(MAX_Y):
        if (output[i][j] < 10):
            print(output[i][j],end='  ')
        elif(output[i][j] < 100):
            print(output[i][j],end=' ')
        elif(output[i][j] < 1000):
            print(output[i][j],end='')
    print()

time = end - start
print("time:",time)
