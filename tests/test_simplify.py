import heterocl as hcl

def test_remove_single_loop():
    hcl.init()
    a = hcl.placeholder((1,))
    b = hcl.compute(a.shape, lambda x: a[x] + 1)
    s = hcl.create_schedule(b)
    ir = hcl.lower(s, [a, b])
    assert "for (x, 0, 1)" not in str(ir)
