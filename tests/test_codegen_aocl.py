import heterocl as hcl

def test_ap_int():
	hcl.init();
	A = hcl.placeholder((1, 32), dtype=hcl.Int(3))
	B = hcl.placeholder((1, 32), dtype=hcl.UInt(3))
	C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], dtype=hcl.Int(8))
	s = hcl.create_schedule([A, B, C])
	code = hcl.build(s, target='aocl')
	assert "ap_int<3>" in code
	assert "ap_uint<3>" in code
	assert "int8" in code 

def test_pragma():
	hcl.init()
	A = hcl.placeholder((10, 32), "A")
	B = hcl.placeholder((10, 32))
	C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j])

	# unroll
	s1 = hcl.create_schedule([A, B, C])
	s1[C].unroll(C.axis[1], factor=4)
	code1 = hcl.build(s1, target='aocl')
	assert "#pragma unroll 4" in code1
	
	# pipeline
	s2 = hcl.create_schedule([A, B, C])
	s2[C].pipeline(C.axis[0], initiation_interval=2)
	code2 = hcl.build(s2, target='aocl')
	assert "#pragma ii 2" in code2

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
    assert "for (int ff_outer = 0; ff_outer < 13; ++ff_outer)" in code
    assert "for (int ff_inner = 0; ff_inner < 5; ++ff_inner)" in code
    assert "if (ff_inner < (64 - (ff_outer * 5)))" in code

if __name__ == '__main__':
    test_ap_int()
    test_pragma()
    test_reorder()
    test_split_fuse()
    test_binary_conv()


