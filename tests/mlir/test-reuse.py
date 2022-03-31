import heterocl as hcl
import numpy as np

def test_conv2D_lb_wb():
    hcl.init()
    A = hcl.placeholder((6, 6), "A")
    rx = hcl.reduce_axis(0, 3, "rx")
    ry = hcl.reduce_axis(0, 3, "ry")
    B = hcl.compute((4, 4), lambda i, j: hcl.sum(A[i+rx, j+ry], axis=[rx, ry]), "B")
    s = hcl.create_schedule([A, B])
    LB = s.reuse_at(A, s[B], B.axis[0])
    WB = s.reuse_at(LB, s[B], B.axis[1])
    s.partition(LB, dim=1)
    s.partition(WB)
    s[B].pipeline(B.axis[1])
    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="csim|csyn", project="conv2d.prj")
    f = hcl.build(s, target)
    np_A = np.zeros((6, 6))
    np_B = np.zeros((4,4))
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    f(hcl_A, hcl_B)

test_conv2D_lb_wb()