import heterocl as hcl
import numpy as np
from itertools import permutations
import os


def test_dataflow_graph():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")
    C = hcl.placeholder((10, 32), "C")
    
    def kernel(A, B, C):
        D = hcl.compute(A.shape, lambda y, x: A[y, x] + B[y, x], "D")
        E = hcl.compute(C.shape, lambda y, x: C[y, x] * D[y, x], "E")
        F = hcl.compute((10, 30), lambda y, x: E[y, x] + E[y, x+1] + E[y, x+2], "F")
        return F

    target = hcl.platform.aws_f1
    # E.reuse.partition is atatched to F
    s = hcl.create_schedule([A, B, C], kernel)
    RB = s.reuse_at(kernel.E, s[kernel.F], kernel.F.axis[1])
    s.partition(RB, hcl.Partition.Block)
    s.partition(kernel.D, hcl.Partition.Block)

    # create super stage for sub-graphs
    s.to([A, B, C], target.xcel)
    s.to(kernel.E, target.host)
    code = str(hcl.lower(s))
    assert "test(A, B, C, E)" in code, code
    print(code)

    # test VHLS and AOCL codegen
    code = str(hcl.build(s, target="vhls"))
    code = str(hcl.build(s, target="aocl"))
    print("Succeed!")

def test_subgraph():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")
    C = hcl.placeholder((10, 32), "C")
    
    def kernel(A, B, C):
        D = hcl.compute(A.shape,  lambda y, x: A[y, x] + B[y, x], "D")
        E = hcl.compute(C.shape,  lambda y, x: C[y, x] * D[y, x], "E")
        F = hcl.compute((10, 30), lambda y, x: E[y, x] + E[y, x+1] + E[y, x+2], "F")
        return F

    target = hcl.platform.aws_f1
    # E.reuse.partition is atatched to F
    s = hcl.create_schedule([A, B, C], kernel)
    RB = s.reuse_at(kernel.E, s[kernel.F], kernel.F.axis[1])
    s.partition(RB, hcl.Partition.Block)
    s.partition(kernel.D, hcl.Partition.Block)

    s.to([A, B, C], target.xcel)
    s.to(kernel.E, target.host)

    # create new sch and return top stage 
    top = s.graph()
    top.dataflow()
    print(hcl.lower(s))

def test_sobel_vivado_hls():
    width, height = 224, 224
    A = hcl.placeholder((height,width,3), "A")
    Gx = hcl.placeholder((3,3),"Gx")
    Gy = hcl.placeholder((3,3),"Gy")

    def sobel(A,Gx,Gy):   
       B = hcl.compute((height,width), lambda x,y: A[x][y][0]+A[x][y][1]+A[x][y][2], "B") 
       r = hcl.reduce_axis(0,3)
       c = hcl.reduce_axis(0,3)
       D = hcl.compute((height-2, width-2), 
            lambda x,y: hcl.sum(B[x+r, y+c]*Gx[r,c], axis=[r,c], name="sum1"), "xx")

       t = hcl.reduce_axis(0, 3)
       g = hcl.reduce_axis(0, 3)
       E = hcl.compute((height-2, width-2), 
            lambda x,y: hcl.sum(B[x+t, y+g]*Gy[t,g], axis=[t,g]), "yy")
       return  hcl.compute((height-2,width-2), 
            lambda x,y:hcl.sqrt(D[x][y]*D[x][y]+E[x][y]*E[x][y])*0.05891867,"Fimg")

    s = hcl.create_schedule([A,Gx,Gy],sobel)
    LBX = s.reuse_at(sobel.B._op, s[sobel.xx], sobel.xx.axis[0], "LBX")
    LBY = s.reuse_at(sobel.B._op, s[sobel.yy], sobel.yy.axis[0], "LBY") 
    WBX = s.reuse_at(LBX, s[sobel.xx], sobel.xx.axis[1], "WBX")
    WBY = s.reuse_at(LBY, s[sobel.yy], sobel.yy.axis[1], "WBY")
    s.partition(LBX)
    s.partition(LBY)
    s.partition(WBX)
    s.partition(WBY)
    s.partition(Gx)
    s.partition(Gy)
    s[sobel.xx].pipeline(sobel.xx.axis[1])
    s[sobel.yy].pipeline(sobel.yy.axis[1])

    target = hcl.platform.zc706 
    s.to([A,Gx,Gy], target.xcel) 
    s.to(sobel.Fimg, target.host)

    target.config(compile="vivado_hls", mode="debug")
    print(hcl.build(s, target))

def test_super_stage():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")
    target = hcl.platform.aws_f1

    def kernel(A, B):
        C = hcl.compute((10, 32), lambda *args : A[args] + B[args], "C")

        with hcl.Stage("Super") as m:
            hcl.update(C, lambda *args: C[args] + 1, "update")

            with hcl.Stage("Plus") as stage:
                with hcl.for_(0, 10) as j:
                    C[j, 0] = 10
        return C

    # place the whole super stage body on device
    def _test_super_stage_on_device():
        s = hcl.create_schedule([A, B], kernel)

        s.to([A, B], target.xcel)
        s.to(kernel.Super.Plus.C, target.host)

        code = str(hcl.lower(s))
        assert "test(C, A, B)" in code
        print("Succeed!")

    # place the whole super stage body on device
    def _test_super_stage_on_device_stream():
        s = hcl.create_schedule([A, B], kernel)

        s.to([A, B], target.xcel, mode=hcl.IO.Stream, depth=10)
        s.to(kernel.Super.Plus.C, target.host, mode=hcl.IO.Stream, depth=10)
        code = str(hcl.lower(s))
        assert "io attr: \"C\" 0 0 1 10" in code
        print("Succeed!")

    # yet to support
    def _test_partial_super_stage_on_device():
        s = hcl.create_schedule([A, B], kernel)
        s.to([A, B], target.xcel)
        s.to(kernel.Super.update.C, target.host)

    _test_super_stage_on_device()
    _test_super_stage_on_device_stream()

def test_inter_kernel_channels():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    C = hcl.placeholder((10, 32), "B")
    def kernel(A, C):
        
        B = hcl.compute((10, 32), lambda *args: 0, "B")
        @hcl.def_([(10, 32), (10, 32)])
        def add(A, B):
            hcl.update(B, lambda *args: A[args] + 1)

        @hcl.def_([(10, 32), (10, 32)])
        def mul(B, C):
            hcl.update(C, lambda *args: B[args] * 2)
            
        add(A, B)
        mul(B, C)
    
    s = hcl.create_schedule([A, C], kernel)
    s.to(kernel.B, s[kernel.mul], s[kernel.add], depth=10)
    code = str(hcl.lower(s))
    print(code)

def test_inter_stage_streaming():
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")

    def kernel(A, B):
        C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], "C")
        D = hcl.compute(C.shape, lambda i, j: C[i][j], "D")
        return D

    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A, B], kernel)
    s.to(kernel.C, s[kernel.D])
    code = str(hcl.lower(s))
    print(code)


if __name__ == '__main__':
    test_inter_kernel_channels()
    test_dataflow_graph()
    test_super_stage()
    test_sobel_vivado_hls()
     #test_subgraph()
