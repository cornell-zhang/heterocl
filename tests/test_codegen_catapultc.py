import heterocl as hcl
import numpy as np

def test_new_platform():
    hcl.init()
    A = hcl.placeholder((10, ), "A")
    B = hcl.compute(A.shape, lambda x: A[x] + 1, "B")

    target = hcl.Platform.asic_hls
    s = hcl.create_schedule([A, B])
    s.to(A, target.xcel, mode=hcl.IO.DMA)
    s.to(B, target.host, mode=hcl.IO.DMA)
    target.config(compile="catapultc", mode="sw_sim")
    f = hcl.build(s, target)
    # print(f)
    
    np_A = np.random.randint(10, size = A.shape)
    np_B = np.zeros(A.shape)
    args = (np_A, np_B)
    f.inspect(args)
    # hcl_A = hcl.asarray(np_A)
    # hcl_B = hcl.asarray(np_B)
    # f(hcl_A, hcl_B)
    f.execute(args)
    

def test_runtime_DMA(): 
    hcl.init()
    A = hcl.placeholder((10, ), "A")
    B = hcl.compute(A.shape, lambda x: A[x] + 1, "B")

    # print(hcl.lower(s))

    target = hcl.Platform.asic_hls
    s = hcl.create_schedule([A, B])
    s.to(A, target.xcel, mode=hcl.IO.DMA)
    s.to(B, target.host, mode=hcl.IO.DMA) 
    target.config(compile="catapultc", mode="debug")
    f = hcl.build(s, target)
    print(f)
    
    # np_A = np.random.randint(10, size = A.shape)
    # np_B = np.zeros(A.shape)
    # hcl_A = hcl.asarray(np_A)
    # hcl_B = hcl.asarray(np_B)
    # f(hcl_A, hcl_B)

def test_runtime_stream(): 
    hcl.init()
    A = hcl.placeholder((10, ), "A")
    B = hcl.compute(A.shape, lambda x: A[x] + 1, "B")

    # print(hcl.lower(s))
 
    
    config = {"host": hcl.dev.asic("mentor"), "xcel": [hcl.dev.asic("mentor")]}
    target = hcl.platform.custom(config)
    target.config(compile="catapultc", mode="sw_sim", backend="catapultc")
    s = hcl.create_schedule([A, B])
    s.to(A, target.xcel, mode=hcl.IO.Stream)
    s.to(B, target.host, mode=hcl.IO.Stream) 
    f = hcl.build(s, target)
    
    np_A = np.random.randint(10, size = A.shape)
    np_B = np.zeros(A.shape)
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    f(hcl_A, hcl_B)

def test_array_add_const():
    hcl.init()
    A = hcl.placeholder((10, ), "A")
    B = hcl.compute(A.shape, lambda x: A[x] + 1, "B")

    s = hcl.create_schedule([A, B])
    # print(hcl.lower(s))

    code = hcl.build(s, target='catapultc')
    print(code)


def test_asic_target():
    hcl.init()
    A = hcl.placeholder((5, 5), "A")
    B = hcl.placeholder((5, 5), "B")

    def kernel(A, B):
        C = hcl.compute(A.shape, lambda i, j: A[i, j] + B[i, j], "C")
        return C

    config = {"host": hcl.dev.asic("mentor"), "xcel": [hcl.dev.asic("mentor")]}
    target = hcl.platform.custom(config)
    s = hcl.create_schedule([A, B], kernel)
    target.config(compile="catapultc", mode="debug", backend="catapultc")

    # config = {
    #   "host" : hcl.dev.cpu("intel", "e5"),
    #   "xcel" : [
    #       hcl.dev.fpga("xilinx", "xcvu19p")
    #   ]
    # }
    # target = hcl.platform.custom(config)
    # # target = hcl.platform.aws_f1
    # s = hcl.create_schedule([A, B], kernel)
    # target.config(compile="vitis", mode="debug", backend="vhls")
    s.to(A, target.xcel, mode=hcl.IO.DMA)
    s.to(B, target.xcel, mode=hcl.IO.DMA)
    s.to(kernel.C, target.host, mode=hcl.IO.DMA)

    code = hcl.build(s, target)
    print(code)

    # np_A  =np.random.randint(10, size = A.shape)
    # np_B  =np.random.randint(10, size = B.shape)
    # hcl_A = hcl.asarray(np_A)
    # hcl_B = hcl.asarray(np_B)
    # hcl_C = hcl.asarray(np.zeros(A.shape))
    # code(hcl_A, hcl_B, hcl_C)

    # np_A = hcl_A.asnumpy()
    # np_B = hcl_B.asnumpy()
    # np_C = hcl_C.asnumpy()

    # print(np_A)
    # print(np_B)
    # print(np_C)


def test_vitis():
    # if os.system("which v++ >> /dev/null") != 0:
    #     return

    def test_arith_ops():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")

        def kernel(A):
            B = hcl.compute(A.shape, lambda *args: A[args] + 1, "B")
            C = hcl.compute(A.shape, lambda *args: B[args] + 1, "C")
            D = hcl.compute(A.shape, lambda *args: C[args] * 2, "D")
            return D

        target = hcl.platform.aws_f1
        s = hcl.create_schedule([A], kernel)
        s.to(kernel.B, target.xcel)
        s.to(kernel.C, target.host)
        target.config(compile="vitis", mode="hw_sim")
        f = hcl.build(s, target)

        np_A = np.random.randint(10, size=(10, 32))
        np_B = np.zeros((10, 32))

        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))
        f(hcl_A, hcl_B)
        ret_B = hcl_B.asnumpy()

        assert np.array_equal(ret_B, (np_A + 2) * 2)

    def test_xrt_stream():
        hcl.init()
        A = hcl.placeholder((10, 32), "A")
        B = hcl.placeholder((10, 32), "B")

        def kernel(A, B):
            C = hcl.compute(A.shape, lambda i, j: A[i, j] + B[i, j], "C")
            D = hcl.compute(C.shape, lambda i, j: C[i, j] + 1, "D")
            return D

        target = hcl.platform.aws_f1
        target.config(compile="vitis", mode="sw_sim")
        s = hcl.create_schedule([A, B], kernel)
        s.to(A, target.xcel, mode=hcl.IO.Stream)
        s.to(B, target.xcel, mode=hcl.IO.DMA)
        s.to(kernel.D, target.host, mode=hcl.IO.Stream)

        f = hcl.build(s, target)
        np_A = np.random.randint(10, size=(10, 32))
        np_B = np.random.randint(10, size=(10, 32))
        np_D = np.zeros((10, 32))

        hcl_A = hcl.asarray(np_A)
        hcl_B = hcl.asarray(np_B, dtype=hcl.Int(32))
        hcl_D = hcl.asarray(np_D)
        f(hcl_A, hcl_B, hcl_D)

        assert np.array_equal(hcl_D.asnumpy(), np_A + np_B + 1)

    test_arith_ops()
    test_xrt_stream()


def test_arithmetic():
    def test_scalar_add():
        hcl.init()
        A = hcl.placeholder((1, ), "A")
        B = hcl.placeholder((1, ), "B")
        C = hcl.placeholder((1, ), "C")

        def simple_add(a, b, c):
            c[0] = a[0] + b[0]

        s = hcl.create_schedule([A, B, C], simple_add)
        print(hcl.lower(s))
        # config = {
        #   "host" : hcl.dev.cpu("intel", "e5"),
        #   "xcel" : [
        #       hcl.dev.fpga("xilinx", "xcvu19p")
        #   ]
        # }
        # p = hcl.platform.custom(config)
        # p.config(compile="vitis", mode="debug", backend="catapultc")
        # s.to([A, B], p.xcel)
        # s.to(C, p.host)

        code = hcl.build(s, target="catapultc")
        print(code)

    def test_scalar_mul():
        hcl.init()
        A = hcl.placeholder((1, ), "A")
        B = hcl.placeholder((1, ), "B")
        C = hcl.placeholder((1, ), "C")

        def simple_mul(a, b, c):
            c[0] = a[0] * b[0]

        s = hcl.create_schedule([A, B, C], simple_mul)

        code = hcl.build(s, target="catapultc")
        print(code)

    def test_scalar_mac():
        hcl.init()
        A = hcl.placeholder((1, ), "A")
        B = hcl.placeholder((1, ), "B")
        C = hcl.placeholder((1, ), "C")
        D = hcl.placeholder((1, ), "D")

        def simple_mac(a, b, c, d):
            d[0] = a[0] + (b[0] * c[0])

        s = hcl.create_schedule([A, B, C, D], simple_mac)

        code = hcl.build(s, target="catapultc")
        print(code)

    test_scalar_add()
    test_scalar_mul()
    test_scalar_mac()


def test_pragma():
    hcl.init()
    A = hcl.placeholder((10, 10), "A")
    B = hcl.placeholder((10, 10), "B")
    C = hcl.compute(A.shape, lambda i, j: A[i][j] + B[i][j])
    # unroll
    s1 = hcl.create_schedule([A, B, C])
    s1[C].unroll(C.axis[1], factor=4)
    code1 = hcl.build(s1, target='catapultc')
    print(code1)

    # pipeline
    s2 = hcl.create_schedule([A, B, C])
    s2[C].pipeline(C.axis[0], initiation_interval=2)
    code2 = hcl.build(s2, target='catapultc')
    print(code2)

    # partition
    # s3 = hcl.create_schedule([A, B, C])
    # s3.partition(A, hcl.Partition.Block, dim=2, factor=2)
    # code3 = hcl.build(s3, target='catapultc')
    # print(code3)


def test_slice():
    hcl.init()

    def test_set_slice():
        A = hcl.placeholder((10, ), "A")

        def kernel(A):
            with hcl.Stage("S"):
                A[0][5:1] = 1

        s = hcl.create_schedule([A], kernel)
        config = {
            "host": hcl.dev.asic("mentor"),
            "xcel": [hcl.dev.asic("mentor")]
        }
        target = hcl.platform.custom(config)
        target.config(compile="catapultc", mode="debug", backend="catapultc")
        code = hcl.build(s, target)
        print(code)

    def test_get_slice():
        A = hcl.placeholder((10, ), "A")

        def kernel(A):
            with hcl.Stage("S"):
                A[0] = A[0][5:1]

        s = hcl.create_schedule([A], kernel)
        config = {
            "host": hcl.dev.asic("mentor"),
            "xcel": [hcl.dev.asic("mentor")]
        }
        target = hcl.platform.custom(config)
        target.config(compile="catapultc", mode="debug", backend="catapultc")
        code = hcl.build(s, target)
        print(code)

    test_set_slice()
    test_get_slice()

def test_stream():
    hcl.init()
    A = hcl.placeholder((10, 32), "A")
    B = hcl.placeholder((10, 32), "B")

    def kernel(A, B):
        C = hcl.compute(A.shape, lambda i, j: A[i, j] + B[i, j], "C")
        D = hcl.compute(C.shape, lambda i, j: C[i, j] + 1, "D")
        return D

    # target = hcl.platform.aws_f1
    # target.config(compile="vitis", mode="debug")
    # s = hcl.create_schedule([A, B], kernel)
    config = {"host": hcl.dev.asic("mentor"), "xcel": [hcl.dev.asic("mentor")]}
    target = hcl.platform.custom(config)
    s = hcl.create_schedule([A, B], kernel)
    target.config(compile="catapultc", mode="debug", backend="catapultc")

    s.to(A, target.xcel, mode=hcl.IO.Stream)
    s.to(B, target.xcel, mode=hcl.IO.Stream)
    s.to(kernel.D, target.host, mode=hcl.IO.Stream)
    code = hcl.build(s, target)

    print(code)


def test_binary_conv():
    hcl.init()
    A = hcl.placeholder((1, 32, 14, 14), dtype=hcl.UInt(1), name="A")
    B = hcl.placeholder((64, 32, 3, 3), dtype=hcl.UInt(1), name="B")
    rc = hcl.reduce_axis(0, 32)
    ry = hcl.reduce_axis(0, 3)
    rx = hcl.reduce_axis(0, 3)
    C = hcl.compute((1, 64, 12, 12),
                    lambda nn, ff, yy, xx: hcl.sum(A[nn, rc, yy + ry, xx + rx]
                                                   * B[ff, rc, ry, rx],
                                                   axis=[rc, ry, rx]),
                    dtype=hcl.UInt(8),
                    name="C")

    s = hcl.create_schedule([A, B, C])
    s[C].split(C.axis[1], factor=5)

    config = {"host": hcl.dev.asic("mentor"), "xcel": [hcl.dev.asic("mentor")]}
    target = hcl.platform.custom(config)
    target.config(compile="catapultc", mode="debug", backend="catapultc")
    s.to(A, target.xcel, mode=hcl.IO.DMA)
    s.to(B, target.xcel, mode=hcl.IO.DMA)
    s.to(C, target.host, mode=hcl.IO.DMA)

    code = hcl.build(s, target)
    print(code)




if __name__ == '__main__':
    test_new_platform()
    # test_runtime_DMA()
    # test_runtime_stream()
    # test_array_add_const()
    # test_asic_target()
    # test_arithmetic()
    # test_pragma()
    # test_slice()
    # test_stream()
    # test_binary_conv()
    
