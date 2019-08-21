import heterocl as hcl


def test_pragma():
	hcl.init()
	A = hcl.placeholder((10, 32), "A")
	B = hcl.placeholder((10, 32))
	C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j])
	# unroll
	s1 = hcl.create_schedule([A, B, C])
	s1[C].unroll(C.axis[1], factor=4)
	code1 = hcl.build(s1, target='aocl')
	code11 = hcl.build(s1, target='sdaccel')
	print (code1)
	assert "#pragma unroll 4" in code1
	print (code11)
	assert "__attribute__((opencl_unroll_hint(4)))" in code11
	# pipeline
	s2 = hcl.create_schedule([A, B, C])
	s2[C].pipeline(C.axis[0], initiation_interval=2)
	code2 = hcl.build(s2, target='aocl')
	code22 = hcl.build(s2, target='sdaccel')
	print (code2)
	assert "#pragma ii 2" in code2
	print (code22)
	assert "__attribute__((xcl_pipeline_loop(2)))" in code22
	# partition
	s3 = hcl.create_schedule([A, B, C])
	s3.partition(A, hcl.Partition.Block, dim=2, factor=2)
	code3 = hcl.build(s3, target='sdaccel')
	print (code3)
	assert "__attribute__((xcl_array_partition(block,2,2)))" in code3





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
    print (code)
    assert "for (ap_int<32>intd_t ff_outer = 0; ff_outer < 13; ++ff_outer)" in code
    assert "for (ap_int<32>intd_t ff_inner = 0; ff_inner < 5; ++ff_inner)" in code
    assert "if (ff_inner < (64 - (ff_outer * 5)))" in code


# def test_partition():
# 	# hcl.init(hcl.Float())
# 	# A = hcl.placeholder((10, 10), "A")
# 	# def kernel(A):
# 	# 	return hcl.compute((8, 8), lambda y, x: A[y][x] + A[y+2][x+2], "B")
# 	# s = hcl.create_schedule(A, kernel)
# 	# s[kernel.B].pipeline(kernel.B.axis[1])
# 	# f = hcl.build(s, target='sdaccel')
# 	# print (f)
# 	hcl.init(hcl.Float())
# 	A = hcl.placeholder((10, 10), "A")
# 	def kernel(A):
# 		return hcl.compute((8, 8), lambda y, x: A[y][x] + A[y+2][x+2], "B")
# 	s = hcl.create_scheme(A, kernel)
# 	s.partition(A)
# 	s[kernel.B].pipeline(kernel.B.axis[1])
# 	f = hcl.build(s, target='sdaccel')
# 	print (f)





if __name__ == '__main__':
	test_pragma()
	test_binary_conv()
