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

        return _make.For(loop_var, _min, extent, node.for_type, node.device_api,
                         body, node.annotate_keys, node.annotate_values)

def register_tensors(tensors):
    for tensor in tensors:
        name = "dummy.update." + tensor.name
        hcl.update(tensor, lambda *args: tensor[args]+1, name)

def include_dependency(files):
    curr_path = os.getcwd()
    if not isinstance(files, list):
        files = [ files ]
    source = list()
    for f in files:
        path = os.path.join(curr_path, f)
        assert os.path.exists(path), path
        source.append(path)
    return source

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
def create_extern_module(stage, ip_type="hls", path=None):

    input_ops   = [i._op for i in stage.input_stages]
    input_bufs  = [i._buf for i in stage.input_stages]
    output_bufs = [stage._buf]

    # input and output arguments
    assert stage.ext_ip_name is not None
    attr_keys, attr_values = ["kname"], [stage.ext_ip_name]
    index = 1
    for tensor in stage.inputs:
        try: 
            attr_keys.append("arg:" + tensor.name)
            v = [ tensor.dtype ] 
            for dim in tensor.shape:
                v.append(str(dim))
            attr_values.append(":".join(v))

        except:
            # input is a scalar data type (constant)
            assert isinstance(tensor, (int, float))
            attr_keys.append("arg:{}".format(tensor))
            shape = "int32" if isinstance(tensor, int) else "float32"
            v = [ shape,  "1" ]
            attr_values.append(":".join(v))

        index += 1

    assert len(stage.source) > 0
    attr_keys.append("source")
    v = []
    for _ in stage.source:
      assert os.path.exists(_), "deps path {} not exists".format(_)
      v.append(_)
    attr_values.append(":".join(v))

    attr_keys.append("port_types")
    if len(stage.port_types) > 0:
        assert len(stage.port_types) == len(stage.inputs)
        v = [ str(int(_)) for _ in stage.port_types ]
        attr_values.append(":".join(v))
    else:
        v = [ str(0) for _ in range(len(stage.inputs)) ]
        attr_values.append(":".join(v))

    op = stage._op.op
    assert ip_type in ["RTL", "HLS", "HOST"]
    print(attr_keys, attr_values)
    body = _make.ExternModule(
        "vhls", _make.StringImm(ip_type), op.body, 
        attr_keys, attr_values)

    new_op = _ExternOp(
        op.name, op.tag, op.axis, 
        input_ops, input_bufs, output_bufs, body)
    stage._op = new_op.output(0)


