"""
Use the Stencil Backend
=======================
**Author**: Yuze Chi (Blaok)

In this tutorial, we show how to use the stencil backend in HeteroCL and
generate HLS C++ code as the result.
"""
import numpy as np

import heterocl as hcl

##############################################################################
# Stencil Comptuation
# -------------------
# Stencil kernels compute output based on a sliding window over the input. The
# following shows an example. It computes the average over a 5-point window.

def jacobi(input_image, output_image):
    def jacobi_kernel(y, x):
        return (input_image[y+1, x-1] +
                input_image[y  , x  ] +
                input_image[y+1, x  ] +
                input_image[y+1, x+1] +
                input_image[y+2, x  ]) / 5

    return hcl.update(output_image, jacobi_kernel, name=output_image.name)

##############################################################################
# Use the Stencil Backend
# -----------------------
# HeteroCL provides a special backend for stencil computation kernels. It can
# be used via the `target` argument when building a program.

dtype = hcl.Float()
input_image = hcl.placeholder((480, 640), name="input", dtype=dtype)
output_image = hcl.placeholder((480, 640), name="output", dtype=dtype)
soda_schedule = hcl.create_schedule([input_image, output_image], jacobi)
soda_schedule[jacobi.output].stencil()
print(hcl.build(soda_schedule, target='soda'))

##############################################################################
# Increase Parallelism
# --------------------
# The above program is written in the SODA DSL, which provides advanced
# optimizations to stencil kernels. One of the optimizations is to provide
# scalable parallelism. To increase parallelism, one can unroll the inner-most
# stencil loop, as follows. The same SODA DSL will be generated, except the
# unroll factor will become 8.

soda_schedule = hcl.create_schedule([input_image, output_image], jacobi)
soda_schedule[jacobi.output].stencil(unroll_factor=8)
print(hcl.build(soda_schedule, target='soda'))

##############################################################################
# Generatel HLS C++ Code
# ----------------------
# The SODA DSL certainly does not compile directly. It needs to be passed to
# the SODA Compiler. HeteroCL provides a built-in target that generates HLS
# C++ code from the intermediate SODA code directly. The generated C++ code is
# valid HLS code and can be passed to HLS vendor tools without modifications.

print(hcl.build(soda_schedule, target='soda_xhls'))
