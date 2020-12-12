import heterocl as hcl
import numpy as np
from hlib.op.extern import *

dtype = hcl.Int()

@register_extern_ip(vendor="xilinx")
def single_fft_hls(X_real, X_imag, F_real=None, F_imag=None, name=None):

    if name is None: name = "fft"
    L = X_real.shape[0]
    assert X_real.shape == X_imag.shape
    assert np.log2(L) % 1 == 0, "length must be power of 2: " + str(L)

    return_tensors = False
    if (F_real is None) and (F_imag is None):
        return_tensors = True
        F_real = hcl.compute((L,), lambda i: 0, name='F_real')
        F_imag = hcl.compute((L,), lambda i: 0, name='F_imag')

    # functional behavior
    with hcl.Stage(name) as Module:
        num_stages = int(np.log2(L))
        bit_width = int(np.log2(L))
        IndexTable = np.zeros((L), dtype='int')
        for i in range(L):
            b = '{:0{width}b}'.format(i, width=bit_width)
            IndexTable[i] = int(b[::-1], 2)

        Table = hcl.copy(IndexTable, "table", dtype=hcl.Int())
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

    Module.ext_ip_name = "fft_wrapper"
    Module.inputs = [ X_real, X_imag, F_real, F_imag, L ]
    Module.source = [ os.path.dirname(os.path.abspath(__file__)) + "/fft.cpp"]

    cmd = "vivado -mode batch -source " + \
        "scripts/gen_xo.tcl -tclargs vadd.xo vadd hw_emu"
    Module.command  = [ cmd ]

    create_extern_module(Module, ip_type="HLS")
    if return_tensors: return F_real, F_imag

