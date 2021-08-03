import heterocl as hcl
import numpy as np
import numpy.testing as tst
import os
import urllib.request
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
    if not os.path.exists("firmware"):
       urllib.request.urlretrieve("https://raw.githubusercontent.com/Hecmay/debug.trace/main/toynn.tar.gz", filename="toynn.tar.gz")
       os.system("tar -zxvf toynn.tar.gz")

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

    target.config(compiler="vitis", mode="debug")
    code = hcl.build(s, target)
    assert "nnet::softmax" in code, code
    os.system("rm -rf firmware toynn.tar.gz")


if __name__ == "__main__":
    test_toy_nn()
