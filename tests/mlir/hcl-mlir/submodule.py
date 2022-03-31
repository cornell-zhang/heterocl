import heterocl as hcl

@hcl.def_([(10,10)])
def submodule(A):
    A = hcl.compute(A.shape, lambda *args : A[args] + 1)