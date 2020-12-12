import heterocl as hcl
import numpy as np
import numpy.testing as tst
import hlib
import os, sys
from itertools import permutations
from hlib.op.extern import create_extern_module, register_extern_ip

dtype = hcl.Float(64)

def test_vecadd_sim(length=32, sim=False):

    if sim is True:
        if os.system("which v++ >> /dev/null") != 0:
            return 

    hcl.init(hcl.Int())
    A = hcl.placeholder((length,), name="A")
    B = hcl.placeholder((length,), name="B")

    def math_func(A, B):
        C = hcl.compute(A.shape, lambda *args: A[args]+1, "C")
        ret = hcl.compute(A.shape, lambda *args: 0, "ret") 
        with hcl.Stage("e") as Module:
            hcl.update(ret, lambda *args: C[args] + B[args])

        Module.ext_ip_name = "vec_add"
        Module.inputs = [ C, B, ret, length ]
        parent = os.path.dirname(os.path.abspath(__file__))
        Module.source = [ os.path.join(parent, "test_ext_ips_source/vadd.cpp") ]

        cmd = "vivado -mode batch -source " + \
            "scripts/gen_xo.tcl -tclargs vadd.xo vadd hw_emu"
        Module.command  = [ cmd ]
        # set up port depth. If the depth is 0, then the port is connected to memory
        # instead of an FIFO. 
        Module.port_types = [256, 0, 0, 0]
        create_extern_module(Module, ip_type="HLS")

        return hcl.compute(A.shape, lambda *args: ret[args] * 2, "out")

    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A, B], math_func)

    # test ir correctness 
    p = hcl.platform.aws_f1
    p.config(compile="vitis", mode="debug")
    code = str(hcl.build(s, p)); print(code)
    pattern = "vec_add(C, B, ret, {});".format(length)
    assert pattern in code

if __name__ == '__main__':
    test_vecadd_sim(32)
