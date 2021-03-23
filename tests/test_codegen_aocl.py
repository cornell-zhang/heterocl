import heterocl as hcl
import tests.__test_codegen_harness as harness
target="aocl"

def test_dtype():
    harness.test_dtype(target, ["int4_t", "uint4_t", "int8_t"], False)

def test_print():
    harness.test_print(target)

def test_pragma():
    harness.test_pragma(target,
                        ["#pragma unroll 4",
                         "#pragma ii 2"],
                        False)

def test_reorder():
	hcl.init()
	A = hcl.placeholder((10, 100), "A")

	def two_stage(A):
		B = hcl.compute(A.shape, lambda x, y : A[x, y] + 1, "B")
		C = hcl.compute(A.shape, lambda x, y : B[x, y] + 1, "C")
		return C

	s = hcl.create_schedule([A], two_stage)
	s_B = two_stage.B
	code = hcl.build(s, target='aocl')
	s[s_B].reorder(s_B.axis[1], s_B.axis[0])
	code2 = hcl.build(s, target='aocl')

def test_split_fuse():
	hcl.init()
	A = hcl.placeholder((10, 100), "A")

	def two_stage(A):
		B = hcl.compute(A.shape, lambda x, y : A[x, y] + 1, "B")
		C = hcl.compute(A.shape, lambda x, y : B[x, y] + 1, 'C')
		return C

	s = hcl.create_schedule([A], two_stage)
	s_B = two_stage.B
	x_out, x_in = s[s_B].split(s_B.axis[0], 5)
	code = hcl.build(s, target='aocl')
	s2 = hcl.create_schedule([A], two_stage)
	s2_B = two_stage.B
	x_y = s[s_B].fuse(s2_B.axis[0], s2_B.axis[1])
	code2 = hcl.build(s2, target='aocl')

def test_binary_conv():
    hcl.init()
    A = hcl.placeholder((1, 32, 14, 14), dtype=hcl.UInt(1), name="A")
    B = hcl.placeholder((64, 32, 3, 3), dtype=hcl.UInt(1), name="B")
    rc = hcl.reduce_axis(0, 32)
    ry = hcl.reduce_axis(0, 3)
    rx = hcl.reduce_axis(0, 3)
    C = hcl.compute((1, 64, 12, 12),
        lambda nn, ff, yy, xx: hcl.sum(
            A[nn, rc, yy + ry, xx + rx] * B[ff, rc, ry, rx], axis=[rc, ry, rx]),
        dtype=hcl.UInt(8), name="C")
    s = hcl.create_schedule([A, B, C])
    s[C].split(C.axis[1], factor=5)
    code = hcl.build(s, target='aocl')
    assert "for (int32_t ff_outer = 0; ff_outer < 13; ++ff_outer)" in code
    assert "for (int32_t ff_inner = 0; ff_inner < 5; ++ff_inner)" in code
    assert "if (ff_inner < (64 - (ff_outer * 5)))" in code

if __name__ == '__main__':
    test_ap_int()
    test_pragma()
    test_reorder()
    test_split_fuse()
    test_binary_conv()
