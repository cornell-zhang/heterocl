import heterocl as hcl
from itertools import permutations

def test_placeholders():

    def move_inputs():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        B = hcl.placeholder((10, 32), "B")
        C = hcl.placeholder((10, 32), "C")
        D = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], "D")
        E = hcl.compute(C.shape, lambda i, j: C[i][j] * D[i][j], "E")
        F = hcl.compute(C.shape, lambda i, j: E[i][j] + 1, "F")

        target = hcl.platform.aws_f1
        s = hcl.create_schedule([A, B, C, D, E, F])
        s.to([A, B, C], target.xcel)
        s.to(E, target.host)
        code = str(hcl.lower(s))
        pattern = "test({}.channel, {}.channel, {}.channel, E.channel)"
        combination = [ pattern.format(*_) for _ in list(permutations(["A", "B", "C"])) ]
        assert any([_ in code for _ in combination])

    def move_outputs():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")

        def kernel(A):
            B = hcl.compute(A.shape, lambda i, j: A[i, j] * 2, "B")
            hcl.update(B, lambda i, j: B[i, j] + 1, "update1")
            hcl.update(B, lambda i, j: B[i, j] * 2, "update2")
            return B

        target = hcl.platform.aws_f1
        s = hcl.create_schedule([A], kernel)
        s.to(A, target.xcel)
        s.to(kernel.update1.B, target.host)

        code = str(hcl.lower(s))
        assert "test(A.channel, B.update.channel)" in code

    def self_move_back():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")

        def kernel(A):
            hcl.update(A, lambda i, j: A[i, j] + 1, "update1")
            hcl.update(A, lambda i, j: A[i, j] * 2, "update2")

        target = hcl.platform.aws_f1
        s = hcl.create_schedule([A], kernel)
        s.to(A, target.xcel)
        s.to(kernel.update1.A, target.host)

        code = str(hcl.lower(s))
        assert "test(A.channel, A.update.channel)" in code

    move_inputs()
    move_outputs()
    self_move_back()


def test_extern_ops():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    def kernel(A):
        B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B")
        C = hcl.compute(A.shape, lambda *args : B[args] + 1, "C")
        D = hcl.compute(A.shape, lambda *args : C[args] * 2, "D")
        return D
    
    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A], kernel)
    s.to(kernel.B, target.xcel)
    s.to(kernel.C, target.host)
    code = str(hcl.lower(s))
    assert "test(B.channel, C.channel)" in code


def test_inner_loops():

    def imperative_loop():
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
        s.to(stage, target.xcel, axis=1)
        code = str(hcl.lower(s))
        pattern = "test({}, {}, {}, {})"
        combination = [ pattern.format(*_) 
            for _ in list(permutations(["A", "B", "C", "i"])) ]
        cond = any([_ in code for _ in combination])
        assert cond, code

    def declarative_loop():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        def kernel(A):
            C = hcl.compute(A.shape, lambda *args : A[args] * 4, "C")
            return C
        
        target = hcl.platform.aws_f1
        s = hcl.create_schedule([A], kernel)
        s.to(kernel.C, target.xcel, axis=1)
        code = str(hcl.lower(s))
        assert "test(C, A, args)" in code 

    def inner_loop_tile():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        def kernel(A):
            C = hcl.compute(A.shape, lambda *args : A[args] * 4, "C")
            return C
        
        target = hcl.platform.aws_f1
        s = hcl.create_schedule([A], kernel)

        stage = kernel.C
        yo, yi = s[stage].split(stage.axis[0], factor=3)
        xo, xi = s[stage].split(stage.axis[1], factor=3)
        s.to(kernel.C, target.xcel, axis=1)
        code = str(hcl.lower(s))
        assert "test(args.outer, C, A)" in code 

    imperative_loop()
    declarative_loop()
    inner_loop_tile() 


def test_kernel():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")
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


def test_inter_stage():
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")

    def kernel(A, B):
        C = hcl.compute(A.shape, 
                lambda i, j: A[i][j] + B[i][j], "C")
        D = hcl.compute(C.shape, 
                lambda i, j: C[i][j], "D")
        return D

    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A, B], kernel)
    s.to(kernel.C, s[kernel.D])
    code = str(hcl.lower(s))
    assert "C.pipe1.write" in code
    assert "C.pipe1.read" in code


def test_extern_op_multicast():
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")

    def kernel(A, B):
        C = hcl.compute(A.shape, 
                lambda i, j: A[i][j] + B[i][j], "C")
        D = hcl.compute(C.shape, 
                lambda i, j: C[i][j] + 1, "D")
        E = hcl.compute(C.shape, 
                lambda i, j: C[i][j] * 2, "E")
        return D, E

    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A, B], kernel)
    s.to(kernel.C, s[kernel.D])
    s.to(kernel.C, s[kernel.E])
    code = str(hcl.lower(s))
    assert "C.pipe1.write" in code
    assert "C.pipe1.read" in code
    assert "C.pipe2.write" in code
    assert "C.pipe2.read" in code


def test_kernel_multicast():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    def kernel(A):
        B = hcl.compute((10, 32), lambda *args: 0, "B")
        C = hcl.compute((10, 32), lambda *args: 0, "C")
        
        @hcl.def_([(10, 32), (10, 32)])
        def add(A, B):
            hcl.update(B, lambda *args: A[args] + 1)

        @hcl.def_([(10, 32), (10, 32)])
        def mul(A, C):
            hcl.update(C, lambda *args: B[args] * 2)
            
        add(A, B)
        mul(A, C)

        D = hcl.compute((10,32), lambda *args: B[args] + C[args], "D")
        return D
    
    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A], kernel)
    # s.to(A, target.xcel)
    # s.to(kernel.D, target.host)
    # s.to(B, s[kernel.mul], s[kernel.add])
    code = str(hcl.lower(s))
    # print(code)
    # assert "test(A.channel, D.channel)" in code


def test_mixed_stream():
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")

    def kernel(A, B):
        C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j], "C")
        D = hcl.compute(C.shape, lambda i, j: C[i][j], "D")
        return D

    target = hcl.platform.aws_f1
    s = hcl.create_schedule([A, B], kernel)
    s.to([A, B], target.xcel)
    s.to(kernel.D, target.host)
    s.to(kernel.C, s[kernel.D])
    code = str(hcl.lower(s))
    pattern = "test({}.channel, {}.channel, D.channel)"
    combination = [ pattern.format(*_)
        for _ in list(permutations(["A", "B"])) ]
    cond = any([_ in code for _ in combination])
    assert cond, code
    assert "C.pipe1.write" in code
    assert "C.pipe1.read" in code


def test_fork_join():

    def inter_stage_fork():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        B = hcl.placeholder((10, 32), "B")

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda i, j: A[i,j] + B[i,j], "C")
            D = hcl.compute(C.shape, lambda i, j: C[i,j] + 1, "D")
            E = hcl.compute(C.shape, lambda i, j: C[i,j] * 2, "E")
            return D, E

        target = hcl.platform.aws_f1
        s = hcl.create_schedule([A, B], kernel)
        s.fork(kernel.C, [kernel.D, kernel.E])
        code = str(hcl.lower(s))
        assert "C.pipe1.write" in code
        assert "C.pipe1.read" in code
        assert "C.pipe2.write" in code

    def inter_stage_join():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        B = hcl.placeholder((10, 32), "B")

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda i, j: 0, "C")
            hcl.update(C, lambda i, j: A[i,j] + 1, "s1")
            hcl.update(C, lambda i, j: B[i,j] * 2, "s2")
            return hcl.compute(C.shape, lambda *args: C[args] + 3, "ret")

        target = hcl.platform.aws_f1
        s = hcl.create_schedule([A, B], kernel)
        s.join([kernel.s1.C, kernel.s2.C], kernel.ret.C)
        code = str(hcl.lower(s))
        assert "C.pipe1.read" in code
        assert "C.pipe2.write" in code

    inter_stage_fork()
    inter_stage_join()

def test_kernel_duplicate():

    def extract_subgraph(combine=False):
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        B = hcl.placeholder((10, 32), "B")

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda i, j: 0, "C")
            hcl.update(C, lambda i, j: A[i,j] + 1, "s1")
            hcl.update(C, lambda i, j: B[i,j] * 2, "s2")
            return hcl.compute(C.shape, lambda *args: C[args] + 3, "ret")

        target = hcl.platform.aws_f1
        s = hcl.create_schedule([A, B], kernel)

        A_, B_ = s.to([A, B], target.xcel)
        ret_ = s.to(kernel.ret, target.host)

        if combine == True:
            # merge the channel stages into 
            s[A_].compute_at(s[B_], 1)
            s[B_].compute_at(s[kernel.C], 1)

            # merge stages from top to bottom 
            s[kernel.C].compute_at(s[kernel.s1], kernel.s1.axis[1])
            s[kernel.s1].compute_at(s[kernel.s2], kernel.s2.axis[1])
            s[kernel.s2].compute_at(s[kernel.ret], kernel.ret.axis[1])

        nodes = s.subgraph(inputs=[A_, B_], outputs=[ret_])
        code = str(hcl.lower(s))

    extract_subgraph(True)


def test_custom_device():

    def custom_target():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        B = hcl.placeholder((10, 32), "B")

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda i, j: A[i,j] + B[i,j], "C")
            D = hcl.compute(C.shape, lambda i, j: C[i,j] + 1, "D")
            return D

        config = {
            "host" : hcl.dev.cpu("intel", "e5"),
            "xcel" : [
                hcl.dev.fpga("xilinx", "xcvu19p")
            ]
        }

        p = hcl.platform.custom(config)
        s = hcl.create_schedule([A, B], kernel)
        s.to(A, p.xcel.hbm[0])
        s.to(B, p.xcel.hbm[1])
        s.to(kernel.D, p.host)
        p.config(compile="vitis", mode="debug", backend="vhls")
        code = hcl.build(s, p)
        assert "MAX_HBM_BANKCOUNT" in code

    custom_target()


if __name__ == '__main__':
    test_placeholders()
    test_extern_ops()
    test_inner_loops()
    test_kernel()
    test_inter_stage()
    test_extern_op_multicast()
    test_kernel_multicast()
    test_mixed_stream()
    test_fork_join()
    test_kernel_duplicate()
    test_custom_device()
