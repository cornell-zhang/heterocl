import heterocl as hcl
import numpy as np
from heterocl import util
from heterocl.tvm import make as _make
from heterocl.tvm import stmt as _stmt
from heterocl.tvm import ir_pass as _pass
from heterocl.tvm._api_internal import _ExternOp
from heterocl.schedule import Schedule, Stage
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


# create hls ip invoked within the top function  
def create_hls_ip(stage, name, args, ip_type="hls", path=None):
    # must be called within a superstage
    input_ops   = [i._op for i in stage.input_stages]
    output_bufs = [stage._buf]


# include external ip files 
def create_extern_module(stage, dicts, ip_type="hls", path=None):
    input_ops   = [i._op for i in stage.input_stages]
    input_bufs  = [i._buf for i in stage.input_stages]
    output_bufs = [stage._buf]

    # input and output arguments
    assert "args" in dicts.keys()
    annotate_dict = dicts
    for name, dtype in dicts["args"]:
        annotate_dict["input::" + name] = dtype 
    del annotate_dict["args"]

    # check dependencies 
    if ("deps" in dicts.keys()):
      assert os.path.exists(dicts["deps"]), \
              "deps path {} not exists".format(dicts["deps"])

    op = stage._op.op
    assert ip_type in ["rtl", "hls", "host"]
    body = _make.ExternModule(
        "top", _make.StringImm(ip_type), op.body, 
        list(annotate_dict.keys()), list(annotate_dict.values()))

    new_op = _ExternOp(
        op.name, op.tag, op.axis, 
        input_ops, input_bufs, output_bufs, body)
    stage._op = new_op.output(0)

