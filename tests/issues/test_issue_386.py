import heterocl as hcl

def test_cast_removal():
    hcl.init()
    
    A = hcl.placeholder((10,10), dtype=hcl.UInt(16), name="A")
    B = hcl.placeholder((10,10), dtype=hcl.Int(16), name="B")

    def algo(A, B):
        def f_mutate(i,j):
            factor = hcl.scalar(B[0][0][13:11], name="factor")
            idx = hcl.scalar(B[0][0][11:0], dtype=hcl.UInt(16), name="idx")
            idx += i * hcl.cast(hcl.UInt(16), factor.v) 
            A[idx][j] = B[idx][j]
        bound = hcl.scalar(5, dtype=hcl.Int(32))
        domain = (hcl.cast(hcl.UInt(32), bound.v), hcl.cast(hcl.UInt(32), bound.v))
        hcl.mutate(domain, f_mutate)
    
    s = hcl.create_schedule([A, B], algo)
    f = hcl.build(s, target="vhls")

if __name__ == '__main__':
    test_cast_removal()