import heterocl as hcl
import numpy as np

def test_partition_before_streaming():
    hcl.init()
    A = hcl.placeholder((10, 10), "A", dtype=hcl.UInt(8))
    def kernel(A):
        B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B", dtype=hcl.UInt(8))
        return B

    target = hcl.Platform.xilinx_zc706
    s = hcl.create_schedule([A], kernel)
    s.partition(A, hcl.Partition.Block, dim=1, factor=2) 
    s.to(A, target.xcel)
    s.to(kernel.B, target.host)
    target.config(compiler="vivado_hls", mode="debug")
    print(hcl.build(s, target))

def test_partition_after_streaming():
    hcl.init()
    A = hcl.placeholder((10, 10), "A", dtype=hcl.UInt(8))
    def kernel(A):
        B = hcl.compute(A.shape, lambda *args : A[args] + 1, "B", dtype=hcl.UInt(8))
        return B

    target = hcl.Platform.xilinx_zc706
    s = hcl.create_schedule([A], kernel)
    s.to(A, target.xcel)
    s.partition(A, hcl.Partition.Block, dim=1, factor=2) # memory optimization
    s.to(kernel.B, target.host)
    target.config(compiler="vivado_hls", mode="debug")
    print(hcl.build(s, target))

if __name__ == '__main__':
    test_partition_before_streaming()
    test_partition_after_streaming()
