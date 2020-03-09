import heterocl as hcl
from itertools import permutations

def test_placeholders():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")
    C = hcl.placeholder((10, 32), "C")
    D = hcl.compute(A.shape, 
            lambda i, j: A[i][j] + B[i][j], "D")
    E = hcl.compute(C.shape, 
            lambda i, j: C[i][j] * D[i][j], "E")
    F = hcl.compute(C.shape, 
            lambda i, j: E[i][j] + 1, "F")

    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A, B, C, D, E, F])
    s.to([A, B, C], target.xcel)
    s.to(E, target.host)
    code = str(hcl.lower(s))
    pattern = "test({}.channel, {}.channel, {}.channel, E.channel)"
    combination = [ pattern.format(*_) for _ in list(permutations(["A", "B", "C"])) ]
    cond = any([_ in code for _ in combination])
    assert cond

def test_extern_ops():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    def kernel(A):
        B = hcl.compute(A.shape, 
                lambda *args : A[args] + 1, "B")
        C = hcl.compute(A.shape, 
                lambda *args : B[args] + 1, "C")
        D = hcl.compute(A.shape, 
                lambda *args : C[args] * 2, "D")
        return D
    
    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A], kernel)
    s.to(kernel.B, target.xcel)
    s.to(kernel.C, target.host)
    code = str(hcl.lower(s))
    assert "test(B.channel, C.channel)" in code
 
def test_loops():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")
    def kernel(A, B):
        C = hcl.compute(A.shape, lambda *args : 0, "C")
        with hcl.Stage("stage"):
            with hcl.for_(0, 10, name="i") as i:
                with hcl.for_(0, 32, name="j") as j:
                    B[i, j] = A[i, j] + B[i, j]
                    C[i, j] = 2 * B[i, j]
        return C
    
    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A, B], kernel)

    stage = kernel.stage
    s.to(stage.i, target.xcel, s[stage])
    code = str(hcl.lower(s))
    assert "test(B, A, i, C)" in code

def test_kernel():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")
    C = hcl.placeholder((10, 32), "C")
    def kernel(A, B):
        
        C = hcl.compute((10, 32), lambda *args: 10)
        @hcl.def_([(10, 32), (10, 32)])
        def add(A, B):
            hcl.update(B, lambda *args: A[args] + 1)

        @hcl.def_([(10, 32), (10, 32)])
        def mul(B, C):
            hcl.update(C, lambda *args: B[args] * 2)
            
        add(A, B)
        mul(B, C)
    
    s = hcl.create_schedule([A, B], kernel)
    s.to(B, s[kernel.mul], s[kernel.add])
    code = str(hcl.lower(s))
    assert "c_buf_1.write" in code
    assert "c_buf_1.read" in code

if __name__ == '__main__':
    test_placeholders()
    test_extern_ops()
    test_loops()
    test_kernel()
