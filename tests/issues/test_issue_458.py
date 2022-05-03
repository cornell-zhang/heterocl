import heterocl as hcl

def test():
    # generates: assert ((uint32)0 < uint32(A[0]));
    # the second uint32 is oddly specified ...
    def func(A):
        hcl.assert_(A[0] > 0, "")

    A = hcl.placeholder((2,), "A", dtype=hcl.UInt(16))
    s = hcl.create_schedule([A], func)
    m = hcl.build(s, "shls")
    print(m)

if __name__ == "__main__":
    test()