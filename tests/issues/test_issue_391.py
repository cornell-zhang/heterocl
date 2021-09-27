import heterocl as hcl
import numpy as np
from PIL import Image
import heterocl as hcl
import numpy as np
import math
import imageio
from urllib.request import urlopen

def test_host_codegen():
    hcl.init(init_dtype=hcl.Float())
    img = Image.open(urlopen('http://i.stack.imgur.com/8zINU.gif'))
    width, height = img.size

    A = hcl.placeholder((height,width,3), "A", dtype=hcl.Float())
    Gx = hcl.placeholder((3,3), "Gx",dtype=hcl.Float())
    Gy = hcl.placeholder((3,3), "Gy",dtype=hcl.Float())

    def sobel(A, Gx, Gy):
        r = hcl.reduce_axis(0,3)
        c = hcl.reduce_axis(0,3)

        A1 = hcl.compute((height,width), lambda y, x: 
            A[y][x][0] + A[y][x][1] + A[y][x][2], "A1")

        B1 = hcl.compute((height-2,width-2), 
                lambda x,y: hcl.sum(A1[x+r,y+c]*Gx[r,c], axis=[r,c], name="sum1"),
                name="B1", dtype=hcl.Float())

        t = hcl.reduce_axis(0,3)
        g = hcl.reduce_axis(0,3)

        B2 = hcl.compute((height-2,width-2), 
                lambda x,y: hcl.sum(A1[x+t,y+g]*Gy[t,g], axis=[t,g], name="sum2"),
                name="B2", dtype=hcl.Float())

        def avg(in1, in2):
            ll = hcl.scalar(in1, "in1")
            lr = hcl.scalar(in2, "in2")
            return hcl.sqrt(ll.v * ll.v + lr.v * lr.v)/4328*255

        return hcl.compute((height-2,width-2), 
                   lambda x, y : avg(B1[x,y], B2[x,y]),
                   name="output", dtype=hcl.Float())

    target = hcl.Platform.aws_f1
    target.config(compiler="vitis", backend="vhls", mode="debug")
    s = hcl.create_schedule([A, Gx, Gy], sobel)

    # Create and partition reuse buffers
    LBX = s.reuse_at(sobel.A1, s[sobel.B1], sobel.B1.axis[0], "LBX")
    LBY = s.reuse_at(sobel.A1, s[sobel.B2], sobel.B2.axis[0], "LBY") 
    WBX = s.reuse_at(LBX, s[sobel.B1], sobel.B1.axis[1], "WBX")
    WBY = s.reuse_at(LBY, s[sobel.B2], sobel.B2.axis[1], "WBY")
    s.partition(LBX, dim=1)
    s.partition(LBY, dim=1)
    s.partition(WBX)
    s.partition(WBY)
    s.partition(Gx)
    s.partition(Gy)

    # Pipeline the loops
    s[sobel.A1].pipeline(sobel.A1.axis[1])
    s[sobel.B1].pipeline(sobel.B1.axis[1])
    s[sobel.B2].pipeline(sobel.B2.axis[1])
    s[sobel.output].pipeline(sobel.output.axis[1])

    # Move inputs to FPGA and output back to CPU
    s.to(sobel.A1, [sobel.B1, sobel.B2])
    s.to([Gx, Gy], target.xcel)
    s.to(sobel.output, target.host)

    # Create FIFO channels between sub-kernels
    s.to(sobel.B1, sobel.output)
    s.to(sobel.B2, sobel.output)
    code = hcl.build(s, target)
    assert "cl::Buffer buffer_A" in code

if __name__ == '__main__':
    test_host_codegen()
