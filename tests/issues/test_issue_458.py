import heterocl as hcl

def test():
    def func(A):
        hcl.assert_(A[0] > 0, "")

    A = hcl.placeholder((2,), "A", dtype=hcl.UInt(16))
    s = hcl.create_schedule([A], func)
    code = hcl.build(s, "shls")
    assert "SC_ASSERT" in code
    assert "uint32" not in code

if __name__ == "__main__":
    test()