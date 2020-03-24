import heterocl as hcl
import numpy as np
from heterocl.tvm import make as _make
from heterocl.tvm._api_internal import _ExternOp
from heterocl.schedule import Schedule
from collections import OrderedDict
dtype = hcl.Int()

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
    body = _make.AttrStmt(
        curr._buf, "test_scope", 
        _make.StringImm("test"), op.body)

    new_op = _ExternOp(
        op.name, op.tag, op.axis, 
        input_ops, input_bufs, output_bufs, body)
    curr._op = new_op.output(0)
    
    ret._tensor = curr._op
    return ret
