import heterocl as hcl
import numpy as np
from hlib.op.extern import create_extern_module, register_extern_ip
import os

dtype = hcl.Int()

@register_extern_ip(vendor="xilinx")
def vadd(A, B, length, ret=None, name=None):

    Len = A.shape[0]
    assert A.shape == B.shape, "shape not match"
    assert Len == length, "shape not match"

    return_tensors = False
    if ret is None:
        ret_name = "ret" if name is None else name
        return_tensors = True
        ret = hcl.compute(A.shape, lambda *args: 0, ret_name) 

    # functional behavior
    with hcl.Stage("vadd") as Module:
        hcl.update(ret, lambda *args: A[args] + B[args])

    Module.inputs = [A, B, ret, length]
    Module.ports  = ["m_axi", "m_axi", "m_axi", "s_axilite"]
    Module.source = [ os.path.dirname(os.path.abspath(__file__)) + "/vadd.cpp"]

    cmd = "vivado -mode batch -source " + \
        "scripts/gen_xo.tcl -tclargs vadd.xo vadd hw_emu"
    Module.command  = [ cmd ]

    create_extern_module(Module, ip_type="HLS")
    if return_tensors: return ret

