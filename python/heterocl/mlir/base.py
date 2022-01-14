from hcl_mlir.build_ir import set_insertion_point
from mlir.ir import *
import hcl_mlir
from mlir.dialects import builtin, std

module = Module.create(hcl_mlir.get_location())
with hcl_mlir.get_context(), hcl_mlir.get_location():
    func = builtin.FuncOp(name="top", type=FunctionType.get(
        inputs=[], results=[]), ip=InsertionPoint(module.body))
    func.add_entry_block()
    set_insertion_point(InsertionPoint(func.entry_block))
print("Done HCL-MLIR initialization")


def get_module():
    return module


def get_function():
    return func


def get_func_body():
    return func.entry_block
