import heterocl as hcl

def test_reuse_before_streaming():
    hcl.init()
    A = hcl.placeholder((10, 10), name="A")
    def kernel(A):
        B = hcl.compute((10, 8), lambda y, x: A[y, x] + A[y, x+1] + A[y, x+2],name="B")
        C = hcl.compute((10, 8), lambda y, x: B[y, x], name="C")
        return C
    s = hcl.create_schedule([A], kernel)
    kernel_B = kernel.B
    RB = s.reuse_at(A, s[kernel_B], kernel_B.axis[1])
    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="csim")
    s.to(kernel.B, target.xcel)
    s.to(kernel.C, target.host)
    f = hcl.build(s, target)

def test_reuse_after_streaming():
    return 
    hcl.init()
    A = hcl.placeholder((10, 10),name="A")
    def kernel(A):
        B = hcl.compute((10, 10), lambda y, x: A[y, x], "B")
        C = hcl.compute((10, 8), lambda y, x: B[y, x] + B[y, x+1] + B[y, x+2], "C")
        return C
    s = hcl.create_schedule([A], kernel)
    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="csim")
    B_ = s.to(kernel.B, target.xcel)
    s.reuse_at(B_, s[kernel.C], kernel.C.axis[1])
    s.to(kernel.C, target.host)
    print(hcl.lower(s))

if __name__ == '__main__':
    test_reuse_before_streaming()
    test_reuse_after_streaming()
