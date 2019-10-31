import heterocl as hcl 
import hlib 

def test_func_print():
    def test_hmm_sim():
        hcl.init()
        x = hcl.placeholder((1,), 'x', dtype=hcl.UInt(64))
        y = hcl.placeholder((64,), 'y', dtype=hcl.UInt(64))
        def kernel(X, Y):
            return hlib.ppac.hmm_sim(X, Y, name='Z')
        s = hcl.create_schedule([x, y], kernel)
        f = hcl.build(s, target='rv64_ppac')
        code = str(f)
        assert 'PPACFunc_HmmSim' in code

    def test_gemm_binary():
        hcl.init()
        data = hcl.placeholder((64, 64), 'd', dtype=hcl.UInt(1))
        weight = hcl.placeholder((64, 64), 'w', dtype=hcl.UInt(1))
        def kernel(d, w):
            return hlib.ppac.gemm_binary(d, w, 'res')
        s = hcl.create_schedule([data, weight], kernel)
        f = hcl.build(s, target='rv64_ppac')
        code = str(f)
        assert 'PPACFunc_GeMMBin' in code

    def test_gemm_multi_bit_unsigned():
        hcl.init()
        data = hcl.placeholder((32, 32), 'd', dtype=hcl.UInt(8))
        weight = hcl.placeholder((32, 32), 'w', dtype=hcl.UInt(8))
        def kernel(d, w):
            return hlib.ppac.gemm_multi_bit(d, w, 'res')
        s = hcl.create_schedule([data, weight], kernel)
        f = hcl.build(s, target='rv64_ppac')
        code = str(f)
        assert 'PPACFunc_GeMMUInt' in code   

    def test_gemm_multi_bit_signed():
        hcl.init()
        data = hcl.placeholder((32, 32), 'd', dtype=hcl.Int(8))
        weight = hcl.placeholder((32, 32), 'w', dtype=hcl.Int(8))
        def kernel(d, w):
            return hlib.ppac.gemm_multi_bit(d, w, 'res')
        s = hcl.create_schedule([data, weight], kernel)
        f = hcl.build(s, target='rv64_ppac')
        code = str(f)
        assert 'PPACFunc_GeMMSInt' in code 
    
    test_hmm_sim()
    test_gemm_binary()
    test_gemm_multi_bit_unsigned()
    test_gemm_multi_bit_signed()

def test_tile():
    def test_hmm_sim():
        hcl.init()
        b_n = 10
        d_n = 256
        X = hcl.placeholder((b_n,), 'X', dtype=hcl.UInt(64))
        Y = hcl.placeholder((d_n,), 'Y', dtype=hcl.UInt(64))
        def kernel(X, Y):
            return hlib.ppac.hmm_sim(X, Y, name='Z')
        s = hcl.create_schedule([X, Y], kernel)
        ir = str(hcl.lower(s))
        assert ('\"_batch_num\"=' + str(b_n)) in ir
        assert ('\"_in_block_num\"=' + str(1)) in ir
        assert ('\"_out_channel_num\"=' + str(d_n)) in ir

    def test_gemm_binary():
        hcl.init()
        b_n, i_c, o_c = 64, 256, 256
        ppac_config = hlib.ppac.PPAC_config(multi_bit=False)
        data = hcl.placeholder((b_n, i_c), 'd', dtype=hcl.UInt(1))
        weight = hcl.placeholder((o_c, i_c), 'w', dtype=hcl.UInt(1))
        def kernel(d, w):
            return hlib.ppac.gemm_binary(d, w, 'res')
        s = hcl.create_schedule([data, weight], kernel)
        ir = str(hcl.lower(s))
        assert ('\"_batch_num\"=' + str(b_n)) in ir
        assert ('\"_in_block_num\"=' + str(i_c // ppac_config.elem_num)) in ir
        assert ('\"_out_channel_num\"=' + str(o_c)) in ir

    def test_gemm_multi_bit():
        hcl.init()
        b_n, i_c, o_c = 64, 256, 256
        ppac_config = hlib.ppac.PPAC_config(multi_bit=True)
        data = hcl.placeholder((b_n, i_c), 'd', dtype=hcl.Int(8))
        weight = hcl.placeholder((o_c, i_c), 'w', dtype=hcl.Int(8))
        def kernel(d, w):
            return hlib.ppac.gemm_multi_bit(d, w, 'res')
        s = hcl.create_schedule([data, weight], kernel)
        ir = str(hcl.lower(s))
        assert ('\"_batch_num\"=' + str(b_n)) in ir
        assert ('\"_in_block_num\"=' + str(i_c // ppac_config.elem_num)) in ir
        assert ('\"_out_channel_num\"=' + str(o_c)) in ir 

    test_hmm_sim()
    test_gemm_binary()
    test_gemm_multi_bit()