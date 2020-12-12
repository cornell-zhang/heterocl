import heterocl as hcl
import numpy as np
from hlib.op.extern import create_extern_module, register_extern_ip
import hlib
import os

def test_vadd_vhls(length):

    hcl.init(hcl.Int())
    A = hcl.placeholder((length,), name="A")
    B = hcl.placeholder((length,), name="B")

    def math_func(A, B):
        ret = hlib.ip.vadd(A, B, length, name="ret")
        return hcl.compute(A.shape, lambda *args: ret[args] * 2, "out")

    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A, B], math_func)
    s.to([A, B, math_func.ret], target.xcel)
    s.to(math_func.vadd.ret, target.host)

    # test ir correctness
    code = str(hcl.build(s, "vhls"))
    with open('vadd.cl', 'w') as f:
        f.write(code)


@register_extern_ip(vendor='xilinx')
def conv2d_nchw_systolic(Tensor, Weight, ret=None, name=None):

     config_in = hcl.compute(Tensor.shape, lambda *args: 1, "cfg_in")
     config_out = hcl.compute(Tensor.shape, lambda *args: 0, "cfg_out")

     return_tensors = False
     if ret is None:
         ret_name = "ret" if name is None else name
         return_tensors = True
         # TODO: calculate output tensor shape
         ret = hcl.compute(Tensor.shape, lambda *args: 0, ret_name)

     with hcl.Stage('conv2d_nchw_systolic') as Module:
         hcl.update(ret, lambda *args: ret[args] + 1)
         hcl.update(Tensor, lambda *args: Tensor[args] + 1)
         hcl.update(Weight, lambda *args: Weight[args] + 1)

     Module.ext_ip_name = "kernel"
     Module.inputs = [Tensor, Weight, ret, config_in, config_out]
     Module.port_types = [1, 1, 1, 1, 1] # FIFO depth (0 indicates FIFO disabled)
     Module.source = [os.path.dirname(os.path.abspath(__file__)) + '/kernel/systolic_array.cpp']

     create_extern_module(Module, ip_type="HLS")
     if return_tensors: return ret

def test_conv2d_nchw_systolic():
    hcl.init(hcl.Float())
    Tensor = hcl.placeholder((1000, 20, 24, 24), name='Input')
    Weight = hcl.placeholder((20, 20, 5, 5), name='Weight')
    def conv2d_nchw(Tensor, Weight):
        return conv2d_nchw_systolic(Tensor, Weight)

    s = hcl.create_schedule([Tensor, Weight], conv2d_nchw)
    p = hcl.platform.aws_f1
    p.config(compile="vitis", mode="debug")

    code = str(hcl.build(s, p))
    print(hcl.lower(s))
    with open('conv2d.cpp', 'w') as f:
        f.write(code)

if __name__ == "__main__":
    test_conv2d_nchw_systolic()
