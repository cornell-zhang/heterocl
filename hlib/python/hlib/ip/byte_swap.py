import heterocl as hcl
import numpy as np
from hlib.op.extern import *

dtype = hcl.Int()

@register_extern_ip(vendor="intel")
def byte_swap_rtl(input_vec, ret=None, name=None):

    if name is None: name = "my_byteswap"

    return_tensors = False
    if ret is None:
        return_tensors = True
        ret = hcl.compute(input_vec.shape, lambda *args: 0, "vec") 

    # functional behavior
    with hcl.Stage("ExternModule") as Module:
        hcl.update(ret, lambda *args:
                input_vec[args] << 16 | input_vec[args] >> 16, "swap")

    dicts = {}
    dicts["name"] = name
    tensors = [input_vec]
    dicts["args"] = [(_.name, _.dtype) for _ in tensors]

    # declare headers and typedef 
    dicts["func"] = """
    my_byteswap({})
"""

    create_extern_module(Module, dicts, ip_type="rtl")
    if return_tensors: return ret

