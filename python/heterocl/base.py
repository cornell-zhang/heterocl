from mlir.ir import *
import hcl_mlir
from mlir.dialects import builtin, std

ctx = Context()
loc = Location.unknown(ctx)
hcl_mlir.register_dialects(ctx)
module = Module.create(loc)
with ctx, loc:
    func = builtin.FuncOp(name="top", type=FunctionType.get(inputs=[], results=[]), ip=InsertionPoint(module.body))
    func.add_entry_block()

def get_context():
    return ctx

def get_loc():
    return loc

def get_module():
    return module

def get_function():
    return func

def get_func_body():
    return func.entry_block