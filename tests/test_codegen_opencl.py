import heterocl as hcl

def test_pragma():
	hcl.init(hcl.Float())
	A = hcl.placeholder((10, 32), "A")
	B = hcl.placeholder((10, 32))
	C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j])
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



if __name__ == '__main__':
	test_pragma()