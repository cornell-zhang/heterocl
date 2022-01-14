import hcl_mlir
from hcl_mlir.build_ir import set_insertion_point

from mlir.dialects import builtin, std
from mlir.ir import *

module = Module.create(hcl_mlir.get_location())
print("Done HCL-MLIR initialization")


def get_module():
    return module


def get_top_function():
    return module.body.operations[0]
