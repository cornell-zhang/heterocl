"""
HeteroCL Tutorial : FPGA Kernel Code Generation
========================================================

**Author**: Cody Hao Yu (hyu@cs.ucla.edu)

This example demonstrates how to use HeteroCL to generate kernel code for
different backend flow to generate FPGA accelerator bistream.

By specifying `target` in the `build` call, HeteroCL will generate the
corresponding kernel code in string representation.
"""

import heterocl as hcl

a = hcl.var("a")
A = hcl.placeholder((10, 10), "A")
B = hcl.compute(A.shape, lambda x, y: A[x, y] * a, "B")

hcl.resize([a, A], hcl.UInt(5))
hcl.resize(B, hcl.UInt(10))

s = hcl.create_schedule(B)

kernel = hcl.build(s, [a, A, B], target='hlsc')
print kernel
