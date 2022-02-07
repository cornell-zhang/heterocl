import heterocl as hcl
import numpy as np
import numpy.testing as tst
import hlib
import os
from hlib.op.extern import (
    create_extern_module, register_extern_ip, 
    register_tensors, include_dependency)

def test_vadd_vhls():

    # Prepare HLS IP
    code_str = """
void vadd(const float* in1, float* in2, int size) {
    for (int j = 0; j < size; j++) {
#pragma HLS PIPELINE II=1
        in2[j] = in1[j] + 1;
    }
}

"""
    with open("vadd.cpp", "w") as fp:
        fp.write(code_str)

    @register_extern_ip(type="vhls")
    def vadd_vhls_ip(op1, op2, size, name=None):
        if name is None: name = "vadd"
        with hcl.Stage("ExternModule.vadd") as Module:
            register_tensors([op1, op2])

        Module.ext_ip_name = name
        Module.inputs = [op1, op2, size]

        # include cpp/hpp files
        deps = os.path.dirname(os.path.abspath(__file__))
        source = [ "vadd.cpp" ]
        Module.source = include_dependency(source)
        create_extern_module(Module, ip_type="HLS")

    dtype = hcl.Float(32)
    hcl.init(dtype)

    size = 1024
    op1 = hcl.placeholder((size,), dtype=dtype, name="op1")
    op2 = hcl.placeholder((size,), dtype=dtype, name="op2")
    def math_func(op1, op2):
        vadd_vhls_ip(op1, op2, size)

    target = hcl.Platform.aws_f1
    s = hcl.create_schedule([op1, op2], math_func)
    s.to(op1, target.xcel)
    s.to(op2, target.host)

    # test ir correctness 
    target.config(compiler="vitis", mode="debug")
    code = hcl.build(s, target)
    assert "#pragma HLS PIPELINE II=1" in code
    os.remove("vadd.cpp")

if __name__ == "__main__":
    test_vadd_vhls()