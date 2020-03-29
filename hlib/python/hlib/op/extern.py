import heterocl as hcl
import numpy as np
from heterocl.tvm import make as _make
from heterocl.tvm._api_internal import _ExternOp
from heterocl.schedule import Schedule
from collections import OrderedDict
dtype = hcl.Int()

def register_extern_ip(**attrs):
    def with_attrs(f):
        for k,v in attrs.items():
            setattr(f, k, v)
        return f
    return with_attrs

# @register_extern_ip(test={"xilinx" : "swswsw"})
def vector_add_rtl(a, b):
    assert a.shape == b.shape
    ret = hcl.compute(a.shape, 
        lambda *args: a[args] + b[args], "vector_add")

    curr = Schedule.last_stages[-1]
    input_ops   = [i._op for i in curr.input_stages]
    input_bufs  = [i._buf for i in curr.input_stages]
    output_bufs = [curr._buf]

    # create new extern op 
    op = ret._tensor.op
    annotate_keys = ["name", "src"]
    annotate_vals = ["vector_add", "test"]
    body = _make.ExternModule(
        "test_scope", 
        _make.StringImm("test"), op.body, 
        annotate_keys, annotate_vals)

    print(body)
    print(vector_add_rtl.test)
    new_op = _ExternOp(
        op.name, op.tag, op.axis, 
        input_ops, input_bufs, output_bufs, body)
    curr._op = new_op.output(0)
    
    ret._tensor = curr._op
    return ret
