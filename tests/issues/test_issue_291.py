import heterocl as hcl
import numpy as np
import numpy.testing as tst
import hlib
import os
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

        ret = hcl.compute(A.shape, lambda *args: 0, "ret") 
        with hcl.Stage("e") as Module:
            hcl.update(ret, lambda *args: A[args] + B[args])

        Module.ext_ip_name = "vec_add"
        Module.inputs = [ A, B, ret, length ]
        Module.source = [ "./test_ext_ips_source/vadd.cpp" ]

        cmd = "vivado -mode batch -source " + \
            "scripts/gen_xo.tcl -tclargs vadd.xo vadd hw_emu"
        Module.command  = [ cmd ]
        create_extern_module(Module, ip_type="HLS")

        return hcl.compute(A.shape, lambda *args: ret[args] * 2, "out")

    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A, B], math_func)

    # test ir correctness 
    p = hcl.platform.aws_f1
    p.config(compile="vitis", mode="debug")
    code = str(hcl.build(s, p))
    pattern = "vec_add(A, B, ret, {});".format(length)
    assert pattern in code

if __name__ == '__main__':
    test_vecadd_sim(32)
