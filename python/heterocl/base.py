from mlir.ir import *
import hcl_mlir

ctx = Context()
loc = Location.unknown(ctx)
hcl_mlir.register_dialects(ctx)
module = Module.create(loc)

def get_context():
    return ctx

def get_loc():
    return loc

def get_module():
    return module