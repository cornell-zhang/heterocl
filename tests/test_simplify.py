import heterocl as hcl

def test_remove_single_loop():
    hcl.init()
    a = hcl.placeholder((1,))
    b = hcl.compute(a.shape, lambda x: a[x] + 1)
    s = hcl.create_schedule([a, b])
    ir = hcl.lower(s)
    assert "for (x, 0, 1)" not in str(ir)

def test_simplify_slice():
    hcl.init()
    A = hcl.placeholder((10,), "A")
    def kernel(A):
        with hcl.Stage():
            A[5][2:2] = 4
    s = hcl.create_schedule(A, kernel)
    ir = hcl.lower(s)
    assert "2:2" not in str(ir)
