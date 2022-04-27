import heterocl as hcl


def test(target="vhls"):

    def kernel(A_name, B_name):
        hcl.init()
        A = hcl.placeholder((1,2,3,4), dtype=hcl.UInt(33), name=A_name)
        B = hcl.compute((1,2,3,4), lambda x,y,z,w: A[x,y,z,w] + 1, name=B_name)
        s = hcl.create_schedule([A, B])
        return s

    def test_space():
        s = kernel("A A", "B B")
        code = hcl.build(s, target=target)
        assert "A A" not in code
        assert "A_A" in code
        assert "B B" not in code
        assert "B_B" in code
    
    def test_slash():
        s = kernel("A/A", "B/B")
        code = hcl.build(s, target=target)
        assert "A/A" not in code
        assert "A_A" in code
        assert "B/B" not in code
        assert "B_B" in code

    def test_hyphen():
        s = kernel("A-A", "B-B")
        code = hcl.build(s, target=target)
        assert "A-A" not in code
        assert "A_A" in code
        assert "B-B" not in code
        assert "B_B" in code

    test_space()
    test_slash()
    test_hyphen()

if __name__ == "__main__":
    test()