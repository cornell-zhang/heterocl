import heterocl as hcl

def test_explicit_unroll():
    hcl.init()
    W = hcl.placeholder((3,), "W")
    X = hcl.placeholder((32,), "X")

    def kernel(W, X):
        k = hcl.reduce_axis(0, 3, "k")
        return hcl.compute((30,), lambda x: hcl.sum(X[x+k]*W[k], axis=k), "Y")
    
    s = hcl.create_schedule([W, X], kernel)
    pes = s.parallel(kernel.Y, axis=kernel.Y.axis[1])
    code = str(hcl.lower(s))
    print(code)

def test_consec_move():
    hcl.init()
    W = hcl.placeholder((3,), "W")
    X = hcl.placeholder((32,), "X")

    def kernel(W, X):
        k = hcl.reduce_axis(0, 3, "k")
        return hcl.compute((30,), lambda x: hcl.sum(X[x+k]*W[k], axis=k), "Y")
    
    s = hcl.create_schedule([W, X], kernel)
    p = hcl.Platform.zc706

    pes = s.parallel(kernel.Y, axis=kernel.Y.axis[1])
    pe1, pe2, pe3 = pes

    # Broadcasting into PEs
    s.to(X, p.xcel).to([pe1, pe2, pe3])
    code = str(hcl.lower(s))
    print(code)

if __name__ == '__main__':
    test_explicit_unroll()
    test_consec_move()