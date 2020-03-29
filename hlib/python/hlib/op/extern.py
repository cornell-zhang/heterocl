import heterocl as hcl
import numpy as np
from heterocl.tvm import make as _make
from heterocl.tvm._api_internal import _ExternOp
from heterocl.schedule import Schedule
from collections import OrderedDict
import os

dtype = hcl.Int()

def register_extern_ip(**attrs):
    def with_attrs(f):
        for k,v in attrs.items():
            setattr(f, k, v)
        return f
    return with_attrs

@register_extern_ip(vendor="xilinx")
def vector_add_rtl(a, b):
    assert a.shape == b.shape
    ret = hcl.compute(a.shape, 
        lambda *args: a[args] + b[args], "vadd_rtl_out")

    curr = Schedule.last_stages[-1]
    input_ops   = [i._op for i in curr.input_stages]
    input_bufs  = [i._buf for i in curr.input_stages]
    output_bufs = [curr._buf]

    # create new extern op 
    name = "rtl_vadd"
    local = os.path.dirname(__file__)
    path  = os.path.abspath(local + \
        "/../../../extern/" + name)
    annotate_dict = {
        "name"     : "rtl_vadd", 
        "binary"   : "$(TEMP_DIR)/vadd.xo",
        "script"   : path + "/scripts",
        "hdl"      : path + "/src/hdl",
        "makefile" : path + "/config.mk",
    }
    # record argument information 
    annotate_dict["input0"]  = a.name
    annotate_dict["input1"]  = b.name
    annotate_dict["output0"] = "vadd_rtl_out"

    op = ret._tensor.op
    body = _make.ExternModule(
        "rtl_ip_core", 
        _make.StringImm("test"), op.body, 
        list(annotate_dict.keys()), list(annotate_dict.values()))

    new_op = _ExternOp(
        op.name, op.tag, op.axis, 
        input_ops, input_bufs, output_bufs, body)
    curr._op = new_op.output(0)
    
    ret._tensor = curr._op
    return ret
