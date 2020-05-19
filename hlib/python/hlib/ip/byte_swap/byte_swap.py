import heterocl as hcl
import numpy as np
from hlib.op.extern import create_extern_module, register_extern_ip
import os

dtype = hcl.Int()

@register_extern_ip(vendor="intel")
def byte_swap_rtl(input_vec, ret=None, name=None):

    if name is None: name = "my_byteswap"

    Len = input_vec.shape[0]
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
    for (int k = 0; k < {}; k++) {{
      vec[k] = my_byteswap({}[k]);
    }}
""".format(Len, input_vec.name)

    # add dependency files or folders
    deps = os.path.dirname(os.path.abspath(__file__))
    dicts["deps"] = deps + "/lib1"

    create_extern_module(Module, dicts, ip_type="rtl")
    if return_tensors: return ret

