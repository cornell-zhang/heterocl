import hcl_mlir

def exp(x):
    return hcl_mlir.MathExpOp(x)

def log(x):
    return hcl_mlir.MathLogOp(x)

def log2(x):
    return hcl_mlir.MathLog2Op(x)

def log10(x):
    return hcl_mlir.MathLog10Op(x)

def sqrt(x):
    return hcl_mlir.MathSqrtOp(x)

def sin(x):
    return hcl_mlir.MathSinOp(x)

def cos(x):
    return hcl_mlir.MathCosOp(x)

def tanh(x):
    return hcl_mlir.MathTanhOp(x)