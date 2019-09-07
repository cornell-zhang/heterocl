import heterocl as hcl

hcl.init(place=hcl.CPU("riscv"))
initiation_interval = 4
a = hcl.placeholder((10, 20), name="a")
b = hcl.placeholder((10, 20), name="b")
c = hcl.placeholder((10, 20), name="c", 
                    place=hcl.FPGA("intel"))
d = hcl.placeholder((10, 20), "d")
e = hcl.placeholder((10, 20), "e")

@hcl.def_([a.shape, b.shape, c.shape])
def ret_add(a, b, c):
    c = hcl.update(c, lambda x, y: a[x, y] + b[x, y], 'c_add')

@hcl.def_([a.shape, b.shape, c.shape])
def ret_mul(a, b, c):
    c = hcl.update(c, lambda x, y: a[x, y] * b[x, y], 'c_mul')

def add_mul(a, b, c, d, e):
  ret_add(a, b, c)
  ret_mul(c, d, e)

# compute customization
s = hcl.create_schedule([a, b, c, d, e], add_mul)
# op1 = add_mul.c_add
# op2 = add_mul.c_mul
# s[op1].pipeline(op1.axis[0], initiation_interval)
s.partition(b, dim=2, factor=2)

# stream into modules / device
print(s[ret_mul])
# s.stream([a, b, d], hcl.FPGA)
# s[c].stream_to(ret_mul)
# s[d].stream_to(hcl.FPGA)

print(hcl.lower(s))
code = hcl.build(s, target="vhls")
print(code)

