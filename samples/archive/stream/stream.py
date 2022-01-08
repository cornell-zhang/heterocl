import heterocl as hcl

hcl.init()
target = hcl.Platform.xilinx_zc706
initiation_interval = 4

a = hcl.placeholder((10, 20), name="a")
b = hcl.placeholder((10, 20), name="b")
c = hcl.placeholder((10, 20), name="c") 
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

# stream into modules / device
a0, b0 = s.to([a, b], target.xcel)
d0 = s.to(d, target.xcel)
#s.partition(b0, dim=2, factor=2)
s.to([a0, b0], s[add_mul.ret_add])
s.to(d0, s[add_mul.ret_mul])

# within device move producer to consumer
s.to(c, s[add_mul.ret_mul],
        s[add_mul.ret_add], depth=10)

# return tensor for inter-device move
# e0 = s.stream_to(e, hcl.CPU('riscv'))

# print(add_mul.ret_mul._buf, c._buf)
print(hcl.lower(s))
code = hcl.build(s, target)
print(code)
# 
# with open("example.cl", "w") as f:
#   f.write(code)
#   f.close()
 
