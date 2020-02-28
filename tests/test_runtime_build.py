import heterocl as hcl
import numpy as np

def test_basic(vivado_hls=False):
    if not vivado_hls: return
    hcl.init()
    target = hcl.platform.zc706
    target.config_tool(compile="vivado_hls")

    A = hcl.placeholder((10,))
    def kernel(A):
        return hcl.compute(A.shape, lambda x: A[x] + 1)
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s, target)

    np_A = np.random.randint(0, 10, A.shape)
    np_B = np.zeros(A.shape)
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    f(hcl_A, hcl_B)
    np_B = np_A + 1
    np.testing.assert_array_equal(np_B, hcl_B.asnumpy())

def test_vitis(vitis=False):
    if not vitis: return
    hcl.init()
    target = hcl.platform.aws_f1
    target.config_tool(compile="vitis")

    A = hcl.placeholder((10,))
    def kernel(A):
        return hcl.compute(A.shape, lambda x: A[x] + 1)
    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s, target)

    np_A = np.random.randint(0, 10, A.shape)
    np_B = np.zeros(A.shape)
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    f(hcl_A, hcl_B)
    np_B = np_A + 1
    np.testing.assert_array_equal(np_B, hcl_B.asnumpy())

