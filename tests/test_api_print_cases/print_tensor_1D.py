# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import numpy as np

hcl.init()

A = hcl.placeholder((10,))


def kernel(A):
    hcl.print(A, "%.0f \0")


s = hcl.create_schedule([A], kernel)
f = hcl.build(s)

np_A = np.random.randint(0, 10, size=(10,))
hcl_A = hcl.asarray(np_A)

f(hcl_A)

s = "["
for i in range(0, 10):
    s += str(np_A[i])
    s += ", "
s += "]"
print(s)
