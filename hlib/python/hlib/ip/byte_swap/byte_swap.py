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

    Module.ext_ip_name = name
    Module.inputs = [ input_vec, ret, Len ]
    Module.source = [ os.path.dirname(os.path.abspath(__file__)) + "/byte_swap.cl"]

    cmd = "cd lib1; " + \
        "aocl library hdl-comp-pkg opencl_lib.xml -o opencl_lib.aoco;" + \
        "aocl library create -name opencl_lib opencl_lib.aoco;"
    Module.command  = [ cmd ]

    create_extern_module(Module, ip_type="RTL")
    if return_tensors: return ret

