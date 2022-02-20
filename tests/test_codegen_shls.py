import heterocl as hcl
import tests.__test_codegen_harness as harness

target="shls"

def test_dtype():
    harness.test_dtype(target, ["sc_int<3>",
                                "sc_uint<3>",
                                "sc_int<8>",
                                "sc_fixed<5, 2>",
                                "sc_ufixed<5, 2>",
                                "sc_fixed<7, 3>"]) 

def test_print():
    harness.test_print(target)

def test_pragma():
    harness.test_pragma(target,
                    ["HLS_UNROLL_LOOP(COMPLETE, 4, \"compute3_j\")",
                     "HLS_PIPELINE_LOOP(HARD_STALL, 2, \"compute3_i\")", 
                     "HLS_MAP_TO_REG_BANK(A)"])