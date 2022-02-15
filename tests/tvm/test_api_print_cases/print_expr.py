import heterocl as hcl
import numpy as np

# case1: int

hcl.init()

A = hcl.placeholder((10,))

def kernel(A):
    hcl.print(A[5])

s = hcl.create_schedule([A], kernel)
f = hcl.build(s)

np_A = np.random.randint(0, 10, size=(10,))
hcl_A = hcl.asarray(np_A)

f(hcl_A)

print(hcl_A.asnumpy()[5])

# case1: uint

hcl.init(hcl.UInt(4))

A = hcl.placeholder((10,))

def kernel(A):
    hcl.print(A[5])

s = hcl.create_schedule([A], kernel)
f = hcl.build(s)

np_A = np.random.randint(20, 30, size=(10,))
hcl_A = hcl.asarray(np_A)

f(hcl_A)

print(hcl_A.asnumpy()[5])

# case3: float

hcl.init(hcl.Float())

A = hcl.placeholder((10,))

def kernel(A):
    hcl.print(A[5], "%.4f\n")

s = hcl.create_schedule([A], kernel)
f = hcl.build(s)

np_A = np.random.rand(10)
hcl_A = hcl.asarray(np_A)

f(hcl_A)

print("%.4f" % hcl_A.asnumpy()[5])

# case4: fixed points

hcl.init(hcl.UFixed(6, 4))

A = hcl.placeholder((10,))

def kernel(A):
    hcl.print(A[5], "%.4f\n")

s = hcl.create_schedule([A], kernel)
f = hcl.build(s)

np_A = np.random.rand(10)
hcl_A = hcl.asarray(np_A)

f(hcl_A)

print("%.4f" % hcl_A.asnumpy()[5])

# case5: two ints

hcl.init()

A = hcl.placeholder((10,))

def kernel(A):
    hcl.print((A[5], A[6]), "%d %d\n")

s = hcl.create_schedule([A], kernel)
f = hcl.build(s)

np_A = np.random.randint(0, 10, size=(10,))
hcl_A = hcl.asarray(np_A)

f(hcl_A)

print(str(np_A[5]) + " " + str(np_A[6]))
