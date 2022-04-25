import heterocl as hcl

def test_complex_select():

    hcl.init(hcl.Int(32))
    a = hcl.placeholder((10,), name="a")
    b = hcl.placeholder((10,), name="b")
    c = hcl.placeholder((10,), name="c")
    d = hcl.placeholder((10,), name="d")
    
    def kernel_select(a, b, c, d):
        use_imm = hcl.scalar(1)
        with hcl.for_(0, 10, name="i") as i:
            src = hcl.select(use_imm == 1, hcl.cast(hcl.Int(16), (c[i] + b[i])), 
                                           hcl.cast(hcl.Int(32), (c[i] - b[i]))
                                           )
            dst = hcl.cast(hcl.Int(32), (2 * (c[i] + b[i])))
            d[i] = hcl.select(dst >= (-1 * src),
                    hcl.select(dst <= src, a[i], src),
                    (-1 * src))
    s = hcl.create_schedule([a, b, c, d], kernel_select)
    code = hcl.build(s, target="vhls")
    assert code.count("?") == 6

if __name__ == "__main__":
    test_complex_select()