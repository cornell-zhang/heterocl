import heterocl as hcl
from heterocl import tvm

def test_complex():
    dtype = hcl.Float()
    A = hcl.placeholder((1,1,8,8),"A",dtype)
    def kernel(A):
        return hcl.compute((1,1,10,10), lambda i, c, x, y: hcl.select(tvm.all(x < 8, y < 8),A[i, c, x, y],0), "B", dtype)
    s = hcl.create_schedule([A], kernel)

    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="debug")
    code = hcl.build(s, target=target)
    assert "A[(c + i)][0][x][y]" in code, code

if __name__ == "__main__":
    test_complex()
