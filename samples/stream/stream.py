import heterocl as hcl

# hcl.init(place=hcl.CPU("riscv"))
hcl.init(place=hcl.FPGA("intel"))
initiation_interval = 4
a = hcl.placeholder((10, 20), name="a")
b = hcl.placeholder((10, 20), name="b")

# auto-alloc empty buffer on fpga 
# c = hcl.placeholder((10, 20), name="c", 
#                     place=hcl.FPGA("intel"))
c = hcl.compute((10, 20), lambda x, y: 0, 
                name = "c")

d = hcl.placeholder((10, 20), name="d")
e = hcl.placeholder((10, 20), name="e")

def add_mul(a, b, c, d, e):
    @hcl.def_([a.shape, b.shape, c.shape])
    def ret_add(a, b, c):
        with hcl.for_(0, a.shape[0]) as i:
            with hcl.for_(0, a.shape[1]) as j:
                c[i, j] = a[i, j] + b[i, j]

    @hcl.def_([c.shape, d.shape, e.shape])
    def ret_mul(c, d, e):
        # hcl.update(c, lambda x, y: a[x, y] * b[x, y], 'c_mul')
        with hcl.for_(0, c.shape[0]) as i:
            with hcl.for_(0, c.shape[1]) as j:
                e[i, j] = c[i, j] * d[i, j]

    ret_add(a, b, c)
    ret_mul(c, d, e)

# compute customization
s = hcl.create_schedule([a, b, c, d, e], add_mul)
# op1 = add_mul.ret_add.c
# op2 = add_mul.ret_mul.c
# s[op1].pipeline(op1.axis[0], initiation_interval)
s.partition(b, dim=2, factor=2)

# stream into modules / device
# a0, b0 = s.stream_to([a, b], hcl.FPGA("intel"))
# s.stream_to([a0, b0], add_mul.ret_add)

# within device move producer to consumer
s[c].stream_to(s[add_mul.ret_add],
               s[add_mul.ret_mul])

# return buffer for inter-device move
d0 = s[d].stream_to(hcl.FPGA('intel'))

# print(add_mul.ret_mul._buf, c._buf)
print(hcl.lower(s))
print(hcl.build(s, target="vhls"))

