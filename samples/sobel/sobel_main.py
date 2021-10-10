"""
Sobel Edge Detection
=====================

**Authors**: YoungSeok Na, Alga Peng, Xiangyi Zhao, Mira Kim

In this example, we will demonstrate Sobel Edge Detection algorithm written in
HeteroCL. This algorithm is a widely used edge detection method used in image
processing. By calculating the image gradient in x- and y-direction, as well as
the gradient magnitude in each pixel, the algorithm attempts to identify pixels
that would correspond to edges in the image.

Additionally, we attempt to demonstrate a HLS-reporting feature to assist
programmers in being able to get a better understading of the performance of
the given program.
"""

import heterocl as hcl
from PIL import Image
import numpy as np
import math
import os
import xmltodict

##############################################################################
# Setup
# =====
# To define a function in HeteroCL, we must define placeholders to create a
# schedule.

DIR = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(DIR, "images/harry.jpg")
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
# we define the reduction axis on both convolution tensors to be a 3x3 region.
# Then, we use the `.compute()` primitive that allows us to compute a new
# tensor based on the function we give it.

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
# buffer to optimize for memory usage. The `reuse_at` primitive supports the
# data reuse in the specified axis. In order to formulate such buffers, we
# need to partition the array via `partition` primitives to achieve II of 1.
# We can also apply `pipeline` primitives to maximize the memory bandwidth via
# loop transformation.

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
# perform the execution. To convert an input to a HeteroCL-compatible type, we
# use `.asarray()` primitive to convert the input into HeteroCL array.
#
# When building the schedule to run the simulation, we use the `hcl.build()` to
# configure the simulation. With only argument being the schedule `s`, we let
# it run a CPU simulation. However, by defining the `target` variable, we can
# tell the HLS tool which hardware it is synthesizing for. This target 
# configuration is required for one to use the HLS reporting feature.

npGx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
npGy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
hcl_Gx = hcl.asarray(npGx)
hcl_Gy = hcl.asarray(npGy)

npF = np.zeros((height-2, width-2))
hcl_F = hcl.asarray(npF)
npA = np.array(img)
hcl_A = hcl.asarray(npA)

if os.system("which vivado_hls >> /dev/null") != 0:
    # CPU simulation                  
    f = hcl.build(s)                  
    f(hcl_A, hcl_Gx, hcl_Gy, hcl_F)   
else:
    # HLS config 
    target = hcl.Platform.xilinx_zc706 
    s.to([A,Gx,Gy], target.xcel) 
    s.to(sobel.Fimg, target.host)
    target.config(compiler="vivado_hls", mode="csyn")
    f = hcl.build(s, target)
    f(hcl_A, hcl_Gx, hcl_Gy, hcl_F)

###############################################################################
# Verification
# ============
# We can verify the result with the ground truth by simply converting the 
# output tensor to a numpy array.

npF = hcl_F.asnumpy()
newimg = np.zeros((height-2, width-2, 3))
for x in range(0, height-2):
    for y in range(0, width-2):
        for z in range(0,3):
            newimg[x,y,z] = npF[x,y]
newimg = newimg.astype(np.uint8)
# imageio.imsave("pic_sobel.jpg",newimg)

###############################################################################
# HLS Report
# =======
# HeteroCL supports an API for report interface that outputs a statistical
# result of resource usage and latency data from the HLS report.

if os.system("which vivado_hls >> /dev/null") != 0:
    xml_file = str(os.path.join(DIR, "images/test_csynth.xml"))
    with open(xml_file, "r") as xml:
        profile = xmltodict.parse(xml.read())["profile"]
    clock_unit = profile["PerformanceEstimates"]["SummaryOfOverallLatency"]["unit"]
    summary = profile["PerformanceEstimates"]["SummaryOfLoopLatency"]
    report = hcl.report.Displayer(clock_unit)
    report.init_table(summary)
    report.collect_data(summary)
else:
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
| Estimated CP      | 8.129 ns                          |
| Latency (cycles)  | Min 270888972; Max 270888972      |
| Interval (cycles) | Min 270858282; Max 270858282      |
| Resources         | Type        Used    Total    Util |
|                   | --------  ------  -------  ------ |
|                   | BRAM_18K       0      280      0% |
|                   | DSP48E        20      220      9% |
|                   | FF          3683   106400      3% |
|                   | LUT         7156    53200     13% |
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
| Latency (cycles)  | Min 4147365; Max 4147365          |
| Interval (cycles) | Min 4147216; Max 4147216          |
| Resources         | Type        Used    Total    Util |
|                   | --------  ------  -------  ------ |
|                   | BRAM_18K      16      280      6% |
|                   | DSP48E       104      220     47% |
|                   | FF         20899   106400     20% |
|                   | LUT        37935    53200     71% |
+-------------------+-----------------------------------+
"""

# For a more detailed analysis of the program, we can employ a "display" API 
# to get information about latency information of the program. To do so,
# simply call ".display()" method on the report. The example output for this
# Sobel example is shown below.

report.display()

"""
TODO: To be updated after FIFO parsing support
Without Optimization:
+-----------+--------------+-----------+---------------------+---------------+------------------+
|           |   Trip Count |   Latency |   Iteration Latency |   Pipeline II |   Pipeline Depth |
|-----------+--------------+-----------+---------------------+---------------+------------------|
| B_x       |          400 |   2240800 |                5602 |           N/A |              N/A |
| + B_y     |          400 |      5600 |                  14 |           N/A |              N/A |
| E_x1      |            3 |       123 |                  41 |           N/A |              N/A |
| + E_y1    |            3 |        39 |                  13 |           N/A |              N/A |
| ++ E_ra2  |          398 |  20751720 |               52140 |           N/A |              N/A |
| +++ E_ra3 |          398 |     52138 |                 131 |           N/A |              N/A |
| D_x3      |            3 |       123 |                  41 |           N/A |              N/A |
| + D_y2    |            3 |        39 |                  13 |           N/A |              N/A |
| ++ D_ra0  |          398 |  20751720 |               52140 |           N/A |              N/A |
| +++ D_ra1 |          398 |     52138 |                 131 |           N/A |              N/A |
| Fimg_x5   |          398 |   4436108 |               11146 |           N/A |              N/A |
| + Fimg_y3 |          398 |     11144 |                  28 |           N/A |              N/A |
+-----------+--------------+-----------+---------------------+---------------+------------------+
* Units in clock cycles

With Optimization:
+-----------------------+--------------+-----------+---------------------+---------------+------------------+
|                       |   Trip Count |   Latency |   Iteration Latency |   Pipeline II |   Pipeline Depth |
|-----------------------+--------------+-----------+---------------------+---------------+------------------|
| B_x_B_y               |       160000 |    320012 |                 N/A |             2 |               15 |
| E_x_reuse_E_y_reuse   |       160000 |    160121 |                 N/A |             1 |              123 |
| D_x_reuse1_D_y_reuse1 |       160000 |    160121 |                 N/A |             1 |              123 |
| Fimg_x3_Fimg_y1       |       158404 |    158431 |                 N/A |             1 |               29 |
+-----------------------+--------------+-----------+---------------------+---------------+------------------+
* Units in clock cycles
"""

# The displayer also supports querying of different parts of the report, be it
# with loop names and/or latency categories. For instance, if you want to query
# the 'Latency' and 'Pipeline II' information of loops 'B' and 'E', we can tell
# the displayer to query only that information. Since it can support multiple
# queries, the arguments must be in a form of a list. It can also take in the
# information regarding the loop-nest depths (loop level) in the program.

report.display(loops=['B', 'E'], cols=['Latency', 'Pipeline II'])

"""
TODO: To be updated after FIFO parsing support
Without Optimization:
+-----------+-----------+---------------+
|           |   Latency |   Pipeline II |
|-----------+-----------+---------------|
| B_x       |   2240800 |           N/A |
| + B_y     |      5600 |           N/A |
| E_x1      |       123 |           N/A |
| + E_y1    |        39 |           N/A |
| ++ E_ra2  |  20751720 |           N/A |
| +++ E_ra3 |     52138 |           N/A |
+-----------+-----------+---------------+
* Units in clock cycles

With Optimization:
+---------------------+-----------+---------------+
|                     |   Latency |   Pipeline II |
|---------------------+-----------+---------------|
| B_x_B_y             |    320012 |             2 |
| E_x_reuse_E_y_reuse |    160121 |             1 |
+---------------------+-----------+---------------+
* Units in clock cycles
"""
