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


# create hls ip invoked within the top function  
def create_hls_ip(op, name, args, ip_type="hls", path=None):
    # must be called within a superstage
    assert Stage._current
    curr = Schedule.last_stages[-1]
    input_ops   = [i._op for i in curr.input_stages]
    output_bufs = [curr._buf]


# include external ip files 
def create_top_module(op, name, args, dicts={}, ip_type="hls", path=None):
    curr = Schedule.last_stages[-1]
    input_ops   = [i._op for i in curr.input_stages]
    input_bufs  = [i._buf for i in curr.input_stages]
    output_bufs = [curr._buf]

    # input and output arguments
    assert len(args) > 0
    annotate_dict = dicts
    for name, dtype in args:
        annotate_dict["input::" + name] = dtype 

    assert ip_type in ["rtl", "hls"]
    body = _make.ExternModule(
        "top", _make.StringImm(ip_type), op.body, 
        list(annotate_dict.keys()), list(annotate_dict.values()))

    new_op = _ExternOp(
        op.name, op.tag, op.axis, 
        input_ops, input_bufs, output_bufs, body)
    curr._op = new_op.output(0)

@register_extern_ip(vendor="xilinx")
def single_fft_hls(X_real, X_imag, F_real=None, F_imag=None, name=None):

    if name is None: name = "hls::fft<config>"
    L = X_real.shape[0]
    assert X_real.shape == X_imag.shape
    assert np.log2(L) % 1 == 0, "length must be power of 2: " + str(L)

    # functional behavior
    with hcl.Stage("ExternModule") as Module:
        num_stages = int(np.log2(L))
        bit_width = int(np.log2(L))
        IndexTable = np.zeros((L), dtype='int')
        for i in range(L):
            b = '{:0{width}b}'.format(i, width=bit_width)
            IndexTable[i] = int(b[::-1], 2)

        return_tensors = False
        Table = hcl.copy(IndexTable, "table", dtype=hcl.Int())
        if (F_real is None) and (F_imag is None):
            return_tensors = True
            F_real = hcl.compute((L,), lambda i: X_real[Table[i]], name='F_real')
            F_imag = hcl.compute((L,), lambda i: X_imag[Table[i]], name='F_imag')
        else: # use passed-in tensors 
            hcl.update(F_real, lambda i: X_real[Table[i]], name='F_real_update')
            hcl.update(F_imag, lambda i: X_imag[Table[i]], name='F_imag_update')

        with hcl.Stage("Out"):
            one = hcl.scalar(1, dtype="int32")
            with hcl.for_(0, num_stages) as stage:
                DFTpts = one[0] << (stage + 1)
                numBF = DFTpts / 2
                e = -2 * np.pi / DFTpts
                a = hcl.scalar(0)
                with hcl.for_(0, numBF) as j:
                    c = hcl.scalar(hcl.cos(a[0]))
                    s = hcl.scalar(hcl.sin(a[0]))
                    a[0] = a[0] + e
                    with hcl.for_(j, L + DFTpts - 1, DFTpts) as i:
                        i_lower = i + numBF
                        temp_r = hcl.scalar(F_real[i_lower] * c - F_imag[i_lower] * s)
                        temp_i = hcl.scalar(F_imag[i_lower] * c + F_real[i_lower] * s)
                        F_real[i_lower] = F_real[i] - temp_r[0]
                        F_imag[i_lower] = F_imag[i] - temp_i[0]
                        F_real[i] = F_real[i] + temp_r[0]
                        F_imag[i] = F_imag[i] + temp_i[0]

    # create module wrapper 
    tensors = [X_real, X_imag, F_real, F_imag]
    args = [(_.name, _.dtype) for _ in tensors]
    # declare headers and typedef 
    config_decl = """
#include \"hls_fft.h\"
#include <complex>
struct config : hls::ip_fft::params_t {
  static const unsigned ordering_opt = hls::ip_fft::natural_order;
  static const unsigned config_width = 16; // FFT_CONFIG_WIDTH
};
typedef std::complex<ap_fixed<16,1>> fxpComplex;
"""
    config_pre_func = """
  hls::ip_fft::config_t<config> fft_config;
  hls::ip_fft::config_t<config> fft_status;
  fft_config.setDir(0);
  fft_config.setSch(0x2AB);
"""
    # interface for extern ip function 
    config_call_func = """
  complex<ap_fixed<16,1>> xn[{}];
  complex<ap_fixed<16,1>> xk[{}];
  for (int i = 0; i < {}; i++) 
    xn[i] = fxpComplex({}[i], {}[i]);
  hls::fft<config>(xn, xk, &fft_config, &fft_status); 
  for (int i = 0; i < {}; i++) {{
    {}[i] = xk.real();
    {}[i] = xk.imag();
  }}
""".format(L, L, L, X_real.name, X_imag.name,
        L, F_real.name, F_imag.name)

    dicts = {}
    dicts["config_decl"] = config_decl
    dicts["config_pre_func"] = config_pre_func
    dicts["config_call_func"] = config_call_func
    create_top_module(Module._op.op, name, args, dicts, ip_type="hls")
    if return_tensors: return F_real, F_imag

