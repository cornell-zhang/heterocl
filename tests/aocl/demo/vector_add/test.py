import heterocl as hcl

hcl.init()

def simple_compute(A,B):
    C = hcl.compute(A.shape, lambda x: A[x]+B[x],"C")
    return C


A = hcl.placeholder((10,),"A")
B = hcl.placeholder((10,),"B")

s = hcl.create_schedule([A,B],simple_compute)
print(hcl.lower(s))
code = hcl.build(s, target="aocl")
print(code)
