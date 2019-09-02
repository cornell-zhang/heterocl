from collections import OrderedDict
import heterocl as hcl
import heterocl.tvm as tvm

class PPAC_config:
    """Wrap PPAC parameters and function names."""
    def __init__(self, multi_bit=False, word_bits=None, elem_bits=None):
        """Initialize PPAC configurations

        Parameters
        ----------
        multi_bit : Whether to use specialized ppac accelerator 
                    or generalized ppac module.
                    See hardware implementation for more.
        word_bits : Number of bits in a row in ppac.
        elem_bits : Number of bits in a number in matrix (datatype)

        """
        self.word_bits = (word_bits if word_bits else 256) if multi_bit else 64
        self.elem_bits = (elem_bits if elem_bits else 8) if multi_bit else 1
        self.elem_num  = self.word_bits // self.elem_bits
        self.depth     = self.elem_num
        assert self.elem_bits in [1, 2, 4, 8, 16, 32], "elem_bits must be in {1, 2, 4, 8, 16, 32}"
        assert (self.word_bits % 64 == 0) and (self.elem_num*self.elem_bits == self.word_bits), \
            "word_bits must be times of 64 and times of elem_bits"
        if multi_bit:
            self.func_call = ['PPACFunc_GeMMUInt', 'PPACFunc_GeMMSInt']
        else:
            self.func_call = ['PPACFunc_HmmSim', 'PPACFunc_GeMMBin']


class PPAC_func_params:
    """
    names of PPAC function call parameters
    used as annotation key on the stage
    """

    def __init__(self):
        self.func_name  = '_ppac_func_name'
        self.ret        = '_ret'
        self.arg0       = '_arg0'
        self.arg1       = '_arg1'
        self.b_n        = '_batch_num'
        self.i_b_n      = '_in_block_num'
        self.o_c_n      = '_out_channel_num'

ppac_params = PPAC_func_params()

def hmm_sim(x, y, name=None):
    """Compute hamming-similarity between each element in x and y
    Parameters
    ----------
    x  : 1-d tensor of datatype uint64
    y  : 1-d tensor of datatype uint64

    Returns
    -------
    res: 2-d tensor of shape (x.shape[0], y.shape[0]) and datatype uint64
     """
    assert x.dtype == 'uint64' and y.dtype == 'uint64', "only support datatype uint64"
    assert len(x.shape) == 1 and len(y.shape) == 1, "only support 1-dim hamming-similarity operation"

    ppac_config = PPAC_config()

    try:
        res_shape = x.shape + y.shape
        batch_num = x.shape[0]
    except:
        # x is scalar
        res_shape = y.shape
        batch_num = 1
    res_name = name if name else 'res'
    in_block_num = 1
    out_channel_num = y.shape[0]

    def _assign_val(*args):
        temp = hcl.local(0, name='sim_acc', dtype=hcl.UInt(64))
        temp[0] = tvm.popcount(~(x[args[0]] ^ y[args[1]]))
        return temp[0]
    return hcl.compute( res_shape, _assign_val, res_name, dtype=hcl.UInt(64),
                        attrs=OrderedDict([(ppac_params.func_name, tvm.make.StringImm(ppac_config.func_call[0])),
                                         (ppac_params.ret,    tvm.make.StringImm(res_name)),
                                         (ppac_params.arg0,   tvm.make.StringImm(x.name)),
                                         (ppac_params.arg1,   tvm.make.StringImm(y.name)),
                                         (ppac_params.b_n,    batch_num),
                                         (ppac_params.i_b_n,  in_block_num),
                                         (ppac_params.o_c_n,  out_channel_num)]) )

def gemm_binary(d, w, name=None):
    """Compute general matrix multiplication of datatype {1, -1}
    Parameters
    ----------
    d  : 2-d tensor of datatype uint1
    w  : 2-d tensor of datatype uint1

    Returns
    -------
    res: 2-d tensor of shape (d.shape[0], w.shape[0]) and datatype uint64
        res = dot(d, w.T) (with datatype {1, -1})
     """
    assert d.dtype == 'uint1' and w.dtype == 'uint1', 'only support binary data'
    assert len(w.shape) == 2 and len(d.shape) == 2, "only support 2-dim binary gemm"
    assert d.shape[1] == w.shape[1]

    ppac_config = PPAC_config()
    assert d.shape[1] % ppac_config.elem_num == 0, \
        "input channel should be times of " + str(ppac_config.elem_num)

    res_name = name if name else 'res'
    batch_num = d.shape[0]
    in_channel_num = w.shape[1]
    in_block_num = in_channel_num // ppac_config.elem_num
    out_channel_num = w.shape[0]
    res_shape = (batch_num, out_channel_num)
    block_size = ppac_config.elem_num // 8

    def _bin_pack_uint8(tensor):
        """Pack uint1 to uint8.
        uint1 is cast to uint8 in c backend.
        This operation squeezes memory 8 times.
        """
        assert tensor.dtype == 'uint1'

        ishape = tensor.shape
        n = len(ishape)
        oshape = ishape[:-1] + (ishape[n-1] // 8, )

        def _assign_val(*args):
            temp = hcl.local(0, name='pack_acc', dtype=hcl.UInt(8))
            with hcl.for_(0, 8) as i:
                temp[0] = temp[0] | (tensor[args[0], i + args[1]*8] << i)
            return temp[0]

        return hcl.compute(oshape, _assign_val,
                           name=tensor.name+'_packed', dtype=hcl.UInt(8))

    def _mvpodd_reduce(*args):
        """compute {1, -1} dot product on packed data."""
        temp = hcl.local(0, name='mvpodd_acc', dtype=hcl.UInt(64))
        with hcl.for_(0, in_block_num) as o:
            with hcl.for_(0, block_size) as i:
                temp[0] += tvm.popcount(d_packed[args[0], i+block_size*o] ^ w_packed[args[1], i+block_size*o])
        temp[0] = ppac_config.elem_num - temp[0]*2
        return temp[0]

    d_packed = _bin_pack_uint8(d)
    w_packed = _bin_pack_uint8(w)
    return hcl.compute(res_shape, _mvpodd_reduce, name=res_name, dtype=hcl.UInt(64),
                       attrs=OrderedDict([(ppac_params.func_name, tvm.make.StringImm(ppac_config.func_call[1])),
                                         (ppac_params.ret,    tvm.make.StringImm(res_name)),
                                         (ppac_params.arg0,   tvm.make.StringImm(d_packed.name)),
                                         (ppac_params.arg1,   tvm.make.StringImm(w_packed.name)),
                                         (ppac_params.b_n,    batch_num),
                                         (ppac_params.i_b_n,  in_block_num),
                                         (ppac_params.o_c_n,  out_channel_num)]) )


def gemm_multi_bit(d, w, name=None):
    """Compute general matrix multiplication of multi-bit data
    Parameters
    ----------
    d  : 2-d tensor
    w  : 2-d tensor

    Returns
    -------
    res: 2-d tensor of shape (d.shape[0], w.shape[0]) and datatype uint64
        res = dot(d, w.T)
     """
    assert w.dtype == d.dtype
    assert w.dtype in ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32']

    assert len(w.shape) == 2 and len(d.shape) == 2, "only support 2-dim gemm"
    assert d.shape[1] == w.shape[1]

    ppac_config = PPAC_config(multi_bit=True)
    assert d.shape[1] % ppac_config.elem_num == 0, \
        "only support data with size of times of " + str(ppac_config.elem_num)

    res_name = name if name else 'res'
    batch_num = d.shape[0]
    in_channel_num = d.shape[1]
    in_block_num = in_channel_num // ppac_config.elem_num
    out_channel_num = w.shape[0]
    res_shape = (batch_num, out_channel_num)
    func_name = ppac_config.func_call[0] if ('u' in d.dtype) else ppac_config.func_call[1]

    r = hcl.reduce_axis(0, in_channel_num, name='k')
    return hcl.compute(res_shape,
                       lambda i, j: hcl.sum(d[i, r] * w[j, r], axis=r),
                       name=res_name, dtype=hcl.UInt(64),
                       attrs=OrderedDict([(ppac_params.func_name, tvm.make.StringImm(func_name)),
                                         (ppac_params.ret,    tvm.make.StringImm(res_name)),
                                         (ppac_params.arg0,   tvm.make.StringImm(d.name)),
                                         (ppac_params.arg1,   tvm.make.StringImm(w.name)),
                                         (ppac_params.b_n,    batch_num),
                                         (ppac_params.i_b_n,  in_block_num),
                                         (ppac_params.o_c_n,  out_channel_num)]))