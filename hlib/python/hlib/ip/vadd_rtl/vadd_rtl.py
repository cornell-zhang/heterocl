import heterocl as hcl
import numpy as np
from hlib.op.extern import create_extern_module, register_extern_ip
import os

dtype = hcl.Int()

@register_extern_ip(vendor="xilinx")
def vadd_rtl(A, B, length, ret=None, name=None):

    if name is None: name = "vadd_rtl"
    Len = A.shape[0]
    assert A.shape == B.shape, "shape not match"
    assert Len == length, "shape not match"

    return_tensors = False
    if ret is None:
        return_tensors = True
        ret = hcl.compute(A.shape, lambda *args: 0, "ret") 

    # functional behavior
    with hcl.Stage("ExternModule") as Module:
        hcl.update(ret, lambda *args:
                A[args] + B[args], "vadd")

    dicts = {}
    dicts["name"] = name
    tensors = [A, B]
    dicts["args"] = [(_.name, _.dtype) for _ in tensors]

    # RTL IP is wrapped as a separate OpenCL kernel in Vitis
    # add dependency files or folders
    # the dependencies are copied to project folder
    deps = os.path.dirname(os.path.abspath(__file__))
    dicts["deps"] = deps + "/scripts"

    # custom compilation command (root path: project) 
    # commands executed before impl or emulation 
    dicts["cmds"] = "vivado -mode batch -source " + \
        "scripts/gen_xo.tcl -tclargs vadd.xo vadd hw_emu {} {}"

    # custom compiler flgas (load custom libs) 
    dicts["flags"] = "vadd.xo"

    create_extern_module(Module, dicts, ip_type="rtl")
    if return_tensors: return ret

