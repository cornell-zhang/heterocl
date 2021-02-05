import heterocl as hcl
import tests.__test_codegen_harness as harness

target="ihls"

def test_dtype():
    harness.test_dtype(target, ["ac_int<3, true>",
                                "ac_int<3, false>",
                                "ac_int<8, true>",
                                "ac_fixed<5, 2, true>",
                                "ac_fixed<5, 2, false>",
                                "ac_fixed<7, 3, true>"])

def test_print():
    harness.test_print(target)

def test_pragma():
    harness.test_pragma(target,
                        ["#pragma unroll 4",
                         "#pragma ii 2"],
                        False)

def test_set_bit():
    harness.test_set_bit(target, "A[0][4] = 1")

def test_set_slice():
    harness.test_set_slice(target, "A[0].set_slc(1, ((ac_int<4, false>)1))")

def test_get_slice():

    A = hcl.placeholder((10,), "A")
    def kernel(A):
        with hcl.Stage("S"):
            A[0] = A[0][5:1]
    s = hcl.create_schedule([A], kernel)
    code = hcl.build(s, target="ihls")
    assert "A[0].slc<4>(1)" in code

