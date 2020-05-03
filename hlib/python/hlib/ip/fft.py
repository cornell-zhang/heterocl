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
from hlib.op.extern import *

dtype = hcl.Int()

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
            one = hcl.scalar(1, dtype="int32", name="one")
            with hcl.for_(0, num_stages) as stage:
                DFTpts = one[0] << (stage + 1)
                numBF = DFTpts / 2
                e = -2 * np.pi / DFTpts
                a = hcl.scalar(0, "a")
                with hcl.for_(0, numBF) as j:
                    c = hcl.scalar(hcl.cos(a[0]), name="cos")
                    s = hcl.scalar(hcl.sin(a[0]), name="sin")
                    a[0] = a[0] + e
                    with hcl.for_(j, L + DFTpts - 1, DFTpts) as i:
                        i_lower = i + numBF
                        temp_r = hcl.scalar(F_real[i_lower] * c - F_imag[i_lower] * s, "temp_r")
                        temp_i = hcl.scalar(F_imag[i_lower] * c + F_real[i_lower] * s, "temp_i")
                        F_real[i_lower] = F_real[i] - temp_r[0]
                        F_imag[i_lower] = F_imag[i] - temp_i[0]
                        F_real[i] = F_real[i] + temp_r[0]
                        F_imag[i] = F_imag[i] + temp_i[0]

    dicts = {}
    dicts["name"] = name
    tensors = [X_real, X_imag, F_real, F_imag]
    dicts["args"] = [(_.name, _.dtype) for _ in tensors]

    # declare headers and typedef 
    dicts["header"] = """
#include \"hls_fft.h\"
#include <complex>
struct config : hls::ip_fft::params_t {
  static const unsigned ordering_opt = hls::ip_fft::natural_order;
  static const unsigned config_width = 16; // FFT_CONFIG_WIDTH
};
typedef std::complex<ap_fixed<16,1>> fxpComplex;
"""
    # extern ip function 
    dicts["func"] = """
  hls::ip_fft::config_t<config> fft_config;
  hls::ip_fft::config_t<config> fft_status;
  fft_config.setDir(0);
  fft_config.setSch(0x2AB);
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

    create_extern_module(Module, dicts, ip_type="hls")
    if return_tensors: return F_real, F_imag

