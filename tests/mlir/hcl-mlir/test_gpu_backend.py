import heterocl as hcl
import numpy as np
import hcl_mlir


def test_host_xcel():
    # Custom GPU platform
    xcel = hcl.devices.GPU("nvidia", "gtx-1080-ti")
    host = hcl.devices.CPU("intel", "e5")

    target = hcl.Platform(
        name = "gpu_platform",
        devs = [host, xcel],
        host = host, xcel = xcel, 
        tool = None
    )
    target.config(compiler="nvcc", project="gpu.prj")

    # vector-add program
    hcl_mlir.enable_extract_function()
    A = hcl.placeholder((256,), "A")
    B = hcl.placeholder((256,), "B")
    def kernel(A, B):
        C = hcl.compute((256,), lambda i: A[i] + B[i], "C")
        return C
    s = hcl.create_schedule([A, B], kernel)

    # thread/block binding
    num_of_threads_per_block = 64
    bx, tx = s[kernel.C].split(kernel.C.axis[0], factor=num_of_threads_per_block)
    s[kernel.C].bind(bx, hcl.BlockIdx.x)
    s[kernel.C].bind(tx, hcl.ThreadIdx.x)

    s.to([A], target.xcel)
    s.to([kernel.C], target.host)
    mod = hcl.build(s, target)


if __name__ == "__main__":
    test_host_xcel()