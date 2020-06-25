import heterocl as hcl

def test_stencil_single_stage():
    A = hcl.placeholder((10, 10), "A")
    def kernel(A):
        return hcl.compute((10, 8), lambda y, x: A[y, x] + A[y, x+1] + A[y, x+2], "B")
    s = hcl.create_schedule(A, kernel)
    s[kernel.B].stencil(burst_width=256, unroll_factor=4)
    ir = str(hcl.lower(s))
    assert "stencil burst_width=256 unroll_factor=4 num_iteration=1" in ir
    assert "inputs=[A]" in ir
    assert "outputs=[B]" in ir

def test_stencil_multi_stage():
    A = hcl.placeholder((10, 10), "A")
    def kernel(A):
        with hcl.Stage("S"):
            B = hcl.compute((10, 8), lambda y, x: A[y, x] + A[y, x+1] + A[y, x+2], "B")
            C = hcl.compute((8, 8), lambda y, x: B[y, x] + B[y+1, x] + B[y+2, x], "C")
    s = hcl.create_schedule(A, kernel)
    s[kernel.S].stencil(burst_width=256, unroll_factor=4)
    ir = str(hcl.lower(s))
    assert "stencil burst_width=256 unroll_factor=4 num_iteration=1" in ir
    assert "inputs=[A]" in ir
    assert "outputs=[C]" in ir

def test_stencil_multi_stencil():
    A = hcl.placeholder((10, 10), "A")
    def kernel(A):
        B = hcl.compute((10, 8), lambda y, x: A[y, x] + A[y, x+1] + A[y, x+2], "B")
        C = hcl.compute((8, 8), lambda y, x: B[y, x] + B[y+1, x] + B[y+2, x], "C")
    s = hcl.create_schedule(A, kernel)
    s[kernel.B].stencil(burst_width=256, unroll_factor=4)
    s[kernel.C].stencil(burst_width=128, unroll_factor=8)
    ir = str(hcl.lower(s))
    assert "stencil burst_width=256 unroll_factor=4 num_iteration=1" in ir
    assert "inputs=[A]" in ir
    assert "outputs=[B]" in ir
    assert "stencil burst_width=128 unroll_factor=8 num_iteration=1" in ir
    assert "inputs=[B]" in ir
    assert "outputs=[C]" in ir

if __name__ == '__main__':
    test_stencil_single_stage()
    test_stencil_multi_stage()
    test_stencil_multi_stencil()
