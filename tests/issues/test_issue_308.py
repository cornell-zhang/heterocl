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

if __name__ == '__main__':
    test_explicit_unroll()