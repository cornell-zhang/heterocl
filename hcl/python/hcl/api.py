import tvm

def var(name = "var", dtype = "int32"):
  return tvm.var(name = name, dtype = dtype)
