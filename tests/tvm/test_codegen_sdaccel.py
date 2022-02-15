import heterocl as hcl

def test_pragma():
    hcl.init(hcl.Float())
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32))
    C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j])
    
    # unroll
    s1 = hcl.create_schedule([A, B, C])
    s1[C].unroll(C.axis[1], factor=6)
    code1 = hcl.build(s1, target='xocl')
    assert "__attribute__((opencl_unroll_hint(6)))" in code1
    
    # pipeline
    s2 = hcl.create_schedule([A, B, C])
    s2[C].pipeline(C.axis[0], initiation_interval=2)
    code2 = hcl.build(s2, target='xocl')
    assert "__attribute__((xcl_pipeline_loop(2)))" in code2
    
    # partition
    s3 = hcl.create_schedule([A, B, C])
    s3.partition(C, hcl.Partition.Block, dim=2, factor=2)
    code3 = hcl.build(s3, target='xocl')
    assert "__attribute__((xcl_array_partition(block,2,2)))" in code3	

if __name__ == "__main__":
    test_pragma()
