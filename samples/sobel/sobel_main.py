"""
Sobel Edge Detection
=====================

**Authors**: Alga Peng, Xiangyi Zhao, YoungSeok Na, Mira Kim

In this example, we will demonstrate Sobel Edge Detection algorithm written in
HeteroCL.
"""

##############################################################################
# Prelude
# =======
# To define a function in HeteroCL, we must define placeholders to create a
# schedule.

import heterocl as hcl
from PIL import Image
import numpy as np
import math

#path = "home.jpg"
path = './examples/images/rose-grayscale.jpg' 
hcl.init(init_dtype=hcl.Float())
img = Image.open(path)
width, height = img.size
Gx = hcl.placeholder((3, 3), "Gx")
Gy = hcl.placeholder((3, 3), "Gy")
A = hcl.placeholder((height, width, 3))

##############################################################################
# Main Algorithm 
# ==============
# We perform a valid convolution with Sobel kernels. To perform convolution, 
# we define the reduction axis on both convolution tensors.

def sobel(A, Gx, Gy):
    B = hcl.compute((height, width), lambda x, y: A[x][y][0] + A[x][y][1] + A[x][y][2], "B")

    r = hcl.reduce_axis(0, 3)
    c = hcl.reduce_axis(0, 3)
    D = hcl.compute((height-2, width-2), 
        lambda x, y: hcl.sum(B[x+r, y+c]*Gx[r, c], axis=[r, c], name = "sum1"), "D")

    t = hcl.reduce_axis(0, 3)
    g = hcl.reduce_axis(0, 3)
    E = hcl.compute((height-2, width-2), 
        lambda x, y: hcl.sum(B[x+t, y+g]*Gy[t, g], axis=[t, g], name = "sum2"), "E")

    # constant factor to normalize the output
    return hcl.compute((height-2,width-2), 
        lambda x, y:hcl.sqrt(D[x][y]*D[x][y]+E[x][y]*E[x][y])*0.05891867, "Fimg")

# create a schedule
s = hcl.create_schedule([A, Gx, Gy], sobel)

###############################################################################
# Optimization
# ============
# HeteroCL provides different primitives to optimize the performance of the
# program. In this example, since convolutions involve overlapped access to
# the same memory location, we can make use of the idea of window- and line-
# buffer to optimize for memory usage. For further optimization, we apply the
# `partition` and `pipeline` primitives to maximize the memory bandwidth.

LBX = s.reuse_at(sobel.B, s[sobel.D], sobel.D.axis[0], "LBX")
LBY = s.reuse_at(sobel.B, s[sobel.E], sobel.E.axis[0], "LBY") 
WBX = s.reuse_at(LBX, s[sobel.D], sobel.D.axis[1], "WBX")
WBY = s.reuse_at(LBY, s[sobel.E], sobel.E.axis[1], "WBY")
s.partition(LBX, dim=1)
s.partition(LBY, dim=1)
s.partition(WBX)
s.partition(WBY)
s.partition(Gx)
s.partition(Gy)
s[sobel.B].pipeline(sobel.B.axis[1])
s[sobel.D].pipeline(sobel.D.axis[1])
s[sobel.E].pipeline(sobel.E.axis[1])
s[sobel.Fimg].pipeline(sobel.Fimg.axis[1])

###############################################################################
# Results
# =======
# We explicitely and numerically define the inputs to the Sobel operation and
# perform the execution.
#
# OPTIONAL: You can specify the FPGA target to run under using the `target`
# variable that you configure and specify in `hcl.build`. If no such device is 
# used, then the `target` argument may be optional.

target = hcl.platform.zc706 
s.to([A,Gx,Gy], target.xcel) 
s.to(sobel.Fimg, target.host)
target.config(compile="vivado_hls", mode="csyn")

npGx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
npGy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
hcl_Gx = hcl.asarray(npGx)
hcl_Gy = hcl.asarray(npGy)

npF = np.zeros((height-2, width-2))
hcl_F = hcl.asarray(npF)
npA = np.array(img)
hcl_A = hcl.asarray(npA)

f = hcl.build(s, target)
f(hcl_A, hcl_Gx, hcl_Gy, hcl_F)

###############################################################################
# HLS Report
# =======
# HeteroCL supports an API for report interface that outputs a statistical
# result of resource usage and latency data from the HLS report.
report = f.report()

# The following shows an example output from the Sobel example laid out in this
# tutorial.

"""
Without Optimization:
+-------------------+-----------------------------------+
| HLS Version       | Vivado HLS 2019.1.3               |
| Product family    | zynq                              |
| Target device     | xc7z020-clg484-1                  |
| Top Model Name    | test                              |
+-------------------+-----------------------------------+
| Target CP         | 10.00 ns                          |
| Estimated CP      | 8.400 ns                          |
| Latency (cycles)  | Min 719636756; Max 719636756      |
| Interval (cycles) | Min 719636757; Max 719636757      |
| Resources         | Type        Used    Total    Util |
|                   | --------  ------  -------  ------ |
|                   | BRAM_18K   12288      280   4389% |
|                   | DSP48E         8      220      4% |
|                   | FF          2539   106400      2% |
|                   | LUT         4756    53200      9% |
+-------------------+-----------------------------------+

With Optimization:
+-------------------+-----------------------------------+
| HLS Version       | Vivado HLS 2019.1.3               |
| Product family    | zynq                              |
| Target device     | xc7z020-clg484-1                  |
| Top Model Name    | test                              |
+-------------------+-----------------------------------+
| Target CP         | 10.00 ns                          |
| Estimated CP      | 9.634 ns                          |
| Latency (cycles)  | Min 10362300; Max 10362300        |
| Interval (cycles) | Min 10362301; Max 10362301        |
| Resources         | Type        Used    Total    Util |
|                   | --------  ------  -------  ------ |
|                   | BRAM_18K   12306      280   4395% |
|                   | DSP48E        48      220     22% |
|                   | FF         11771   106400     11% |
|                   | LUT        20059    53200     38% |
+-------------------+-----------------------------------+

"""

# For a more detailed analysis of the program, we can employ a "display" API 
# to get information about latency information of the program. To do so,
# simply call ".display()" method on the report. The example output for this
# Sobel example is shown below.
report.display()

"""
+-----------------------+--------------+-----------+---------------------+---------------+------------------+
|                       |   Trip Count |   Latency |   Iteration Latency |   Pipeline II |   Pipeline Depth |
|-----------------------+--------------+-----------+---------------------+---------------+------------------|
| B_x_B_y               |      2073600 |   4147214 |                 N/A |             2 |               17 |
| D_x_reuse_D_y_reuse   |      2073600 |   2073722 |                 N/A |             1 |              124 |
| E_x_reuse1_E_y_reuse1 |      2073600 |   2073722 |                 N/A |             1 |              124 |
| Fimg_x3_Fimg_y1       |      2067604 |   2067634 |                 N/A |             1 |               32 |
+-----------------------+--------------+-----------+---------------------+---------------+------------------+
* Units in clock cycles
"""

# The displayer also supports querying of different parts of the report, be it
# with loop names and/or latency categories. For instance, if you want to query
# the 'Latency' and 'Pipeline II' information of loops 'B' and 'E', we can tell
# the displayer to query only that information. Since it can support multiple
# queries, the  arguments must be in a form of a list.
report.display(loops=['B', 'E'], cols=['Latency', 'Pipeline II'])

"""
+-----------------------+-----------+---------------+
|                       |   Latency |   Pipeline II |
|-----------------------+-----------+---------------|
| B_x_B_y               |   4147214 |             2 |
| E_x_reuse1_E_y_reuse1 |   2073722 |             1 |
+-----------------------+-----------+---------------+
* Units in clock cycles
"""
