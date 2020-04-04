import heterocl as hcl
import numpy as np
from heterocl import util
from heterocl.tvm import make as _make
from heterocl.tvm import stmt as _stmt
from heterocl.tvm import ir_pass as _pass
from heterocl.tvm._api_internal import _ExternOp
from heterocl.schedule import Schedule
from heterocl.mutator import Mutator
from collections import OrderedDict
import os

dtype = hcl.Int()

class ModuleMarker(Mutator):
    """ create extern module used at inner-loop level"""
    def __init__(self, axis, info, args):
        self.axis = axis
        self.info = info
        self.args = args
        self.count = 0
        self.range_ = {}
        self.index_map = {}

    def record_index(self, var, index):
        for key, value in self.range_.items():
          if value == 1:
            sub = {key : 0}
            index = _pass.Simplify(_pass.Substitute(index, sub))
        self.index_map[var] = index

    def mutate_Load(self, node):
        buffer_var = self.mutate(node.buffer_var)
        index = self.mutate(node.index)
        index = util.CastRemover().mutate(index)
        self.record_index(buffer_var, index)

        predicate = self.mutate(node.predicate)
        return _make.Load(node.dtype, buffer_var, index, predicate)

    def mutate_Store(self, node):
        buffer_var = self.mutate(node.buffer_var)
        index = self.mutate(node.index)
        index = util.CastRemover().mutate(index)
        self.record_index(buffer_var, index)

        value = self.mutate(node.value)
        predicate = self.mutate(node.predicate)
        return _make.Store(buffer_var, value, index, predicate)

    def mutate_For(self, node):
        self.count += 1
        loop_var = self.mutate(node.loop_var)
        _min = self.mutate(node.min)
        extent = self.mutate(node.extent)
        self.range_[loop_var] = extent.value - _min.value
        body = self.mutate(node.body)

        if (self.count == self.axis):
            self.count = 0
            if isinstance(body, _stmt.AttrStmt):
                body = body.body
            # insert index map
            index_map = { k.name : v for k, v in self.index_map.items() }
            for i in range(len(self.args)):
                self.info["index" + str(i)] = str(index_map[self.args[i]])

            body = _make.ExternModule(
                "rtl", 
                _make.StringImm("test"), body, 
                list(self.info.keys()), list(self.info.values()))

        return _make.For(loop_var, _min, extent, node.for_type, node.device_api, body)


def register_extern_ip(**attrs):
    def with_attrs(f):
        for k,v in attrs.items():
            setattr(f, k, v)
        return f
    return with_attrs


@register_extern_ip(vendor="xilinx")
def vector_add_rtl(a, b, name=None):

    if name is None: name = "vadd_rtl_out"
    assert a.shape == b.shape
    ret = hcl.compute(a.shape, 
        lambda *args: a[args] + b[args], name)

    curr = Schedule.last_stages[-1]
    input_ops   = [i._op for i in curr.input_stages]
    input_bufs  = [i._buf for i in curr.input_stages]
    output_bufs = [curr._buf]

    # create new extern op 
    func_name = "rtl_vadd"
    local = os.path.dirname(__file__)
    path  = os.path.abspath(local + \
        "/../../../extern/" + func_name)
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
    annotate_dict["output0"] = name

    op = ret._tensor.op
    body = _make.ExternModule(
        "rtl", 
        _make.StringImm("test"), op.body, 
        list(annotate_dict.keys()), list(annotate_dict.values()))

    new_op = _ExternOp(
        op.name, op.tag, op.axis, 
        input_ops, input_bufs, output_bufs, body)
    curr._op = new_op.output(0)
    
    ret._tensor = curr._op
    return ret

@register_extern_ip(vendor="xilinx")
def scalar_add_rtl(a, b, name=None):

    if name is None: name = "rtl_model"
    assert a.shape == b.shape
    ret = hcl.compute(a.shape, 
            lambda *args: a[args] + b[args], name)

    curr = Schedule.last_stages[-1]
    input_ops   = [i._op for i in curr.input_stages]
    input_bufs  = [i._buf for i in curr.input_stages]
    output_bufs = [curr._buf]
    op = ret._tensor.op

    axis = len(curr.axis)
    local = os.path.dirname(__file__)
    path  = os.path.abspath(local + \
        "/../../../extern/" + "rtl_sadd")
    annotate_dict = {
        "name"     : "rtl_model", 
        "json"     : path + "/rtl_model.json", 
        "spec"     : path + "/rtl_model.cpp", 
        "decl"     : "void rtl_model(ap_int<32> a, ap_int<32> b, ap_int<32> &z1);"
    }
    annotate_dict["input0"]  = a.name
    annotate_dict["input1"]  = b.name
    annotate_dict["output0"] = name  

    args = [a.name, b.name, name]
    mutator = ModuleMarker(axis, annotate_dict, args)
    body = mutator.mutate(op.body)

    new_op = _ExternOp(
        op.name, op.tag, op.axis, 
        input_ops, input_bufs, output_bufs, body)
    curr._op = new_op.output(0)
    
    ret._tensor = curr._op
    return ret
