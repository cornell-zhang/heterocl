import heterocl as hcl

hcl.init()
initiation_interval = 4
a = hcl.placeholder((10, 20))
b = hcl.placeholder((10, 20))

@hcl.def_([a.shape, b.shape, (), ()])
def ret_add(a, b, x, y):
    hcl.return_(a[x, y] + b[x, y])

@hcl.def_([a.shape, b.shape, (), ()])
def ret_mul(a, b, x, y):
    hcl.return_(a[x, y] * b[x, y])

c = hcl.compute(a.shape, lambda i, j: ret_add(a, b, i, j))
d = hcl.compute(b.shape, lambda i, j: ret_mul(a, b, i, j))
s = hcl.create_schedule([a, b, c, d])

# compute customization
s[c].pipeline(c.axis[0], initiation_interval)
s.partition(b, dim=2, factor=2)

# stream into modules / device
# s[c].stream_to(ret_mul)
# s[d].stream_to(hcl.FPGA)

print(hcl.lower(s))
code = hcl.build(s, target="vhls")
print(code)


