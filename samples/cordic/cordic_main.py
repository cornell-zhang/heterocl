"""
HeteroCL Tutorial : CORDIC Design
=================================

**Author**: Yi-Hsiang Lai (seanlatias@github)

COordinate Rotation DIgital Computer (CORDIC) is a method for calculating a
variety of functions including trigonometric and hyperbolic. The various
functions are calculated through an iterative set of vector rotations. At the
end of these rotations, the value of the function is easily determined from
the (x, y) coordinate. A CORDIC is often used to achieve low-cost
multiplierless sine/cosine implementations in FPGA as well as ASIC designs.

In this tutorial, we demonstrate how to make use of the decoupled quantization
schemes and algorithms in HeteroCL. We also show how we can explore different
quantization schemes with the quantize API.
"""
# Import modules and set constants.
import heterocl as hcl
import numpy as np
import math
import os

cordic_ctab = [
        0.78539816339744828000,0.46364760900080609000,0.24497866312686414000,
        0.12435499454676144000,0.06241880999595735000,0.03123983343026827700,
        0.01562372862047683100,0.00781234106010111110,0.00390623013196697180,
        0.00195312251647881880,0.00097656218955931946,0.00048828121119489829,
        0.00024414062014936177,0.00012207031189367021,0.00006103515617420877,
        0.00003051757811552610,0.00001525878906131576,0.00000762939453110197,
        0.00000381469726560650,0.00000190734863281019,0.00000095367431640596,
        0.00000047683715820309,0.00000023841857910156,0.00000011920928955078,
        0.00000005960464477539,0.00000002980232238770,0.00000001490116119385,
        0.00000000745058059692,0.00000000372529029846,0.00000000186264514923,
        0.00000000093132257462,0.00000000046566128731,0.00000000023283064365,
        0.00000000011641532183,0.00000000005820766091,0.00000000002910383046,
        0.00000000001455191523,0.00000000000727595761,0.00000000000363797881,
        0.00000000000181898940,0.00000000000090949470,0.00000000000045474735,
        0.00000000000022737368,0.00000000000011368684,0.00000000000005684342,
        0.00000000000002842171,0.00000000000001421085,0.00000000000000710543,
        0.00000000000000355271,0.00000000000000177636,0.00000000000000088818,
        0.00000000000000044409,0.00000000000000022204,0.00000000000000011102,
        0.00000000000000005551,0.00000000000000002776,0.00000000000000001388,
        0.00000000000000000694,0.00000000000000000347,0.00000000000000000173,
        0.00000000000000000087,0.00000000000000000043,0.00000000000000000022]

K_const = 0.6072529350088812561694

##############################################################################
# Main Algorithm
# ==============
# We let the data type be the input argument of our top function. This is how
# we can set different quantization schemes.
def cordic(X, Y, C, theta, N):

    # Prepare all input values and intermediate variables.
    T = hcl.compute((1,), lambda x: 0, "T", X.dtype)
    current = hcl.compute((1,), lambda x: 0, "current", X.dtype)

    # Main loop body: The more steps we iterate, the better accuracy we get.
    def step_loop(step):
        with hcl.if_(theta[0] > current[0]):
            T[0] = X[0] - (Y[0] >> step)
            Y[0] = Y[0] + (X[0] >> step)
            X[0] = T[0]
            current[0] = current[0] + C[step]
        with hcl.else_():
            T[0] = X[0] + (Y[0] >> step)
            Y[0] = Y[0] - (X[0] >> step)
            X[0] = T[0]
            current[0] = current[0] - C[step]

    # This is the main computation that calls the loop body.
    hcl.mutate((N,), lambda step: step_loop(step), "calc")


###############################################################################
# Test with Different Data Types
# ==============================
#
# Set the range of the angle we want to test and set the number of iterations.
NUM = 90
_N = 60
from cordic_golden import golden

###############################################################################
# Loop through different bit-widths.
for b in range(2, 64, 4):

    dtype = hcl.Fixed(b, b-2)
    hcl.init(dtype)

    X = hcl.placeholder((1,), "X")
    Y = hcl.placeholder((1,), "Y")
    C = hcl.placeholder((63,), "cordic_ctab")
    theta = hcl.placeholder((1,), "theta")
    N = hcl.placeholder((), "N", hcl.Int(32))

    s = hcl.create_schedule([X, Y, C, theta, N], cordic)
    f = hcl.build(s)

    acc_err_sin = 0.0
    acc_err_cos = 0.0

    # Loop for testing different angles.
    for d in range(1, NUM):

        _d = math.radians(d)
        ms = math.sin(_d)
        mc = math.cos(_d)

        _X = hcl.asarray(np.array([K_const]))
        _Y = hcl.asarray(np.array([0]))
        _C = hcl.asarray(np.array(cordic_ctab))
        _theta = hcl.asarray(np.array([_d]))

        f(_X, _Y, _C, _theta, _N)

        _X = _X.asnumpy()
        _Y = _Y.asnumpy()

        # We calculate the RMS error.
        err_ratio_sin = math.fabs((ms - _Y[0])/ms) * 100
        err_ratio_cos = math.fabs((mc - _X[0])/mc) * 100

        acc_err_sin += err_ratio_sin * err_ratio_sin
        acc_err_cos += err_ratio_cos * err_ratio_cos

    str_err_sin = str(math.sqrt(acc_err_sin/(NUM-1)))
    str_err_cos = str(math.sqrt(acc_err_cos/(NUM-1)))
    print(str(dtype) + ": " + str_err_sin + " " + str_err_cos)

    index = (b-2) // 4
    assert np.allclose(float(str_err_sin), golden[index][0])
    assert np.allclose(float(str_err_cos), golden[index][1])
