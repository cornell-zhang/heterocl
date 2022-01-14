import hcl_mlir
from mlir.ir import *

module = Module.create(hcl_mlir.get_location())
print("Done HCL-MLIR initialization")


def get_module():
    return module


def get_top_function():
    return module.body.operations[0]
