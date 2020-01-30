import heterocl as hcl
import hlib

hcl.init()
target = hcl.platform.aws_f1
initiation_interval = 4

a = hcl.placeholder((10, 20), name="a")
b = hcl.placeholder((10, 20), name="b")

def add_mul(A, B):
    hlib.function.sort(A, B)

# compute customization
s = hcl.create_schedule([a, b], add_mul)

s.to(a, target.xcel)
s.to(b, target.host)

# print(add_mul.ret_mul._buf, c._buf)
print(hcl.lower(s))
code = hcl.build(s, target)
print(code)
# 
# with open("example.cl", "w") as f:
#   f.write(code)
#   f.close()
 
