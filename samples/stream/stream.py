import heterocl as hcl

# hcl.init(place=hcl.CPU("riscv"))
hcl.init(place=hcl.FPGA("intel"))
initiation_interval = 4
a = hcl.placeholder((10, 20), name="a")
b = hcl.placeholder((10, 20), name="b")
c = hcl.placeholder((10, 20), name="c", 
                    place=hcl.FPGA("intel"))
d = hcl.placeholder((10, 20), "d")
e = hcl.placeholder((10, 20), "e")

def add_mul(a, b, c, d, e):
    @hcl.def_([a.shape, b.shape, c.shape])
    def ret_add(a, b, c):
        with hcl.for_(0, a.shape[0]) as i:
            with hcl.for_(0, a.shape[1]) as j:
                c[i, j] = a[i, j] + b[i, j]

    @hcl.def_([a.shape, b.shape, c.shape])
    def ret_mul(a, b, c):
        # hcl.update(c, lambda x, y: a[x, y] * b[x, y], 'c_mul')
        with hcl.for_(0, a.shape[0]) as i:
            with hcl.for_(0, a.shape[1]) as j:
                c[i, j] = a[i, j] * b[i, j]

    ret_add(a, b, c)
    ret_mul(c, d, e)

# compute customization
s = hcl.create_schedule([a, b, c, d, e], add_mul)
# op1 = add_mul.ret_add.c
# op2 = add_mul.ret_mul.c
# s[op1].pipeline(op1.axis[0], initiation_interval)
s.partition(b, dim=2, factor=2)

# stream into modules / device
# s.stream([a, b], add_mul.ret_add)
s[c].stream_to(s[add_mul.ret_mul])
# s[d].stream_to(hcl.FPGA)

print(hcl.lower(s))
code = hcl.build(s, target="vhls")
print(code)

