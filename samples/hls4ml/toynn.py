import heterocl as hcl
import numpy as np
import numpy.testing as tst
import hlib
import os
from itertools import permutations
from hlib.op.extern import (
    create_extern_module, register_extern_ip, 
    register_tensors, include_dependency)

@register_extern_ip(type="vhls")
def toynn_vhls_ip(input_1, output_1, name=None):
    if name is None: name = "myproject"
    # Function behavior definition
    with hcl.Stage("ExternModule.toyNN") as Module:
        register_tensors([input_1, output_1])

    Module.ext_ip_name = name
    Module.inputs = [input_1, output_1]

    # Include cpp/hpp files
    deps = os.path.dirname(os.path.abspath(__file__))
    source = [
        "firmware/myproject.cpp",
        "firmware/nnet_utils/",
        "firmware/weights/"
    ]
    Module.source = include_dependency(source)
    create_extern_module(Module, ip_type="HLS")


def test_toy_nn():
    dtype = hcl.Float(32)
    hcl.init(dtype)

    input_1 = hcl.placeholder((16,), dtype=dtype, name="input_1")
    output_1 = hcl.placeholder((5,), dtype=dtype, name="output_1")
    def math_func(input_1, output_1):
        toynn_vhls_ip(input_1, output_1)

    target = hcl.Platform.aws_f1
    s = hcl.create_schedule([input_1, output_1], math_func)
    s.to(input_1, target.xcel)
    s.to(output_1, target.host)

    # test ir correctness 
    ir = str(hcl.lower(s))
    print(ir)

    target.config(compiler="vitis", mode="debug")
    code = hcl.build(s, target)
    print(code)

    # hcl_in = hcl.asarray(np_A)
    # hcl_out = hcl.asarray(np_B)

    # hcl_out = hcl.asarray(np.zeros((length)))
    # f(hcl_A, hcl_B, hcl_out)
    # np.testing.assert_array_equal(np_out, hcl_out.asnumpy())

if __name__ == "__main__":
    test_toy_nn()
