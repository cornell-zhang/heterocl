import heterocl as hcl

def test_dtype(target, strings, test_fixed_point=True):
    def test_int():
        hcl.init()
        A = hcl.placeholder((1, 32), dtype=hcl.Int(3))
        B = hcl.placeholder((1, 32), dtype=hcl.UInt(3))
        C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], dtype=hcl.Int(8))
        s = hcl.create_schedule([A, B, C])
        code = hcl.build(s, target=target)
        assert strings[0] in code
        assert strings[1] in code
        assert strings[2] in code

    def test_fixed():
        hcl.init()
        A = hcl.placeholder((1, 32), dtype=hcl.Fixed(5, 3))
        B = hcl.placeholder((1, 32), dtype=hcl.UFixed(5, 3))
        C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], dtype=hcl.Fixed(7, 4))
        s = hcl.create_schedule([A, B, C])
        code = hcl.build(s, target=target)
        assert strings[3] in code
        assert strings[4] in code
        assert strings[5] in code

    test_int()
    if test_fixed_point:
        test_fixed()

def test_print(target):
    hcl.init()
    A = hcl.placeholder((10, 32))
    def kernel(A):
        hcl.print(A[0])
        return hcl.compute(A.shape, lambda *args: A[args])
    s = hcl.create_schedule([A], kernel)
    code = hcl.build(s, target=target)

def test_pragma(target, strings, test_partition=True):
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32))
    C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j])
    # unroll
    s1 = hcl.create_schedule([A, B, C])
    s1[C].unroll(C.axis[1], factor=4)
    code1 = hcl.build(s1, target=target)
    assert strings[0] in code1
    # pipeline
    s2 = hcl.create_schedule([A, B, C])
    s2[C].pipeline(C.axis[0], initiation_interval=2)
    code2 = hcl.build(s2, target=target)
    assert strings[1] in code2
    if test_partition:
        # partition
        s3 = hcl.create_schedule([A, B, C])
        s3.partition(A, hcl.Partition.Block, dim=2, factor=2)
        code3 = hcl.build(s3, target=target)
        assert strings[2] in code3

def test_set_bit(target, string):
    hcl.init()
    A = hcl.placeholder((10,), "A")
    def kernel(A):
        with hcl.Stage("S"):
            A[0][4] = 1
    s = hcl.create_schedule([A], kernel)
    code = hcl.build(s, target=target)
    assert string in code

def test_set_slice(target, string):
    hcl.init()
    A = hcl.placeholder((10,), "A")
    def kernel(A):
        with hcl.Stage("S"):
            A[0][5:1] = 1
    s = hcl.create_schedule([A], kernel)
    code = hcl.build(s, target=target)
    assert string in code

