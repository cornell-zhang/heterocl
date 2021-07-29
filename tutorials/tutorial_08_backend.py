"""
Back-end Support
================

**Author**: Yi-Hsiang Lai (seanlatias@github)

HeteroCL provides multiple back-end supports. Currently, we support both CPU
and FPGA flows. We will be extending to other back ends including ASICs and
PIMs (processing in memory). To set to different back ends, simply set the
``target`` of ``hcl.build`` API. In this tutorial, we will demonstrate how
to target different back ends in HeteroCL. The same program and schedule will
be used throughout the entire tutorial.
"""
import heterocl as hcl
import numpy as np

hcl.init()
A = hcl.placeholder((10, 10), "A")
def kernel(A):
    return hcl.compute((8, 8), lambda y, x: A[y][x] + A[y+2][x+2], "B")
s = hcl.create_scheme(A, kernel)
s.downsize(kernel.B, hcl.UInt(4))
s = hcl.create_schedule_from_scheme(s)
s.partition(A)
s[kernel.B].pipeline(kernel.B.axis[1])
##############################################################################
# CPU
# ---
# CPU is the default back end of a HeteroCL program. If you want to be more
# specific, set the ``target`` to be ``llvm``. Note the some customization
# primitives are ignored by the CPU back end. For instance, ``partition`` and
# ``pipeline`` have no effect. Instead, we can use ``parallel``.
f = hcl.build(s) # equivalent to hcl.build(s, target="llvm")

##############################################################################
# We can execute the returned function as we demonstrated in other tutorials.
hcl_A = hcl.asarray(np.random.randint(0, 10, A.shape))
hcl_B = hcl.asarray(np.zeros((8, 8)), dtype=hcl.UInt(4))
f(hcl_A, hcl_B)

##############################################################################
# FPGA
# ----
# For FPGA, we provide several back ends.
#
# Vivado HLS C++ Code Generation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# To generate Vivado HLS code, simply set the target to ``vhls``. Note that
# the returned function is a **code** instead of an executable.
f = hcl.build(s, target="vhls")
print(f)

##############################################################################
# Vivado HLS C++ Code Simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HeteroCL provides users with the ability to simulation the generated HLS
# code directly from the Python interface. To use this feature, you need to
# have the Vivado HLS header files in your ``g++`` include path. If this is
# the case, then we can set target to ``vhls_csim``, which returns an
# **executable**. We can then run it the same as what we do for the CPU back
# end.
#
# .. note::
#
#    The Vivado HLS program will not be triggered during the simulation.
#    We only need the header files to be in the path.
import subprocess
import sys
proc = subprocess.Popen(
        "g++ -E -Wp,-v -xc++ /dev/null",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        )
stdout, stderr = proc.communicate()
if "Vivado_HLS" in str(stderr):
    f = hcl.build(s, target="vhls_csim")
    f(hcl_A, hcl_B)

##############################################################################
# Intel HLS C++ Code Generation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HeteroCL can also generate Intel HLS code. However, due to certain
# limitation, some directives cannot be generated. To generate the code, set
# the target to ``ihls``.
f = hcl.build(s, target="ihls")
print(f)

##############################################################################
# SODA Stencil Code Generation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HeteroCL incorporates the SODA framework for efficient stencil architecture
# generation. For more details, please refer to
# :ref:`sphx_glr_tutorials_tutorial_09_stencil.py`.
