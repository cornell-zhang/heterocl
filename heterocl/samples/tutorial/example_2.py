"""
HeteroCL Tutorial : FPGA Kernel Code Generation
========================================================

**Author**: Cody Hao Yu (hyu@cs.ucla.edu)

This example demonstrates how to use HeteroCL to generate kernel code for
different backend flow to generate FPGA accelerator bistream.

By specifying `target` in the `build` call, HeteroCL will generate the
corresponding kernel code in string representation. Current available
backend targets are: MerlinC (merlinc).
"""

import heterocl as hcl
import numpy as np

a = hcl.var("a")
A = hcl.placeholder((10,), name="A")
B = hcl.compute((10,), [A], lambda x: A[x] * a, name="B")

hcl.resize([a, A], "uint5")
hcl.resize(B, "uint10")

s = hcl.create_schedule(B)

kernel = hcl.build(s, [a, A, B], target='merlinc')
print kernel
