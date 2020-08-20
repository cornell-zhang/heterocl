import heterocl as hcl
import numpy as np

hcl.init()

def gemm(m=8, n=8, k=8, dtype=hcl.Int(), target=None):
    matrix_1 = hcl.placeholder((m, k), dtype=dtype)
    matrix_2 = hcl.placeholder((k, n), dtype=dtype)

    def kernel(matrix_1, matrix_2):
        r = hcl.reduce_axis(0, k, 'k')
        return hcl.compute((m, n),
                lambda x, y: hcl.sum(matrix_1[x, r] * matrix_2[r, y],
                                     axis=r, dtype=dtype),
                dtype=dtype,
                name="out_matrix")

    s = hcl.create_schedule([matrix_1, matrix_2], kernel)
    
    target = hcl.platform.zc706
    s.to([matrix_1, matrix_2], target.xcel)
    s.to(kernel.out_matrix, target.host)
    
    target.config(compile="vivado_hls", mode="csim|csyn")
    
    hcl_m1 = hcl.asarray(np.random.randint(10, size=(m, k)), dtype=dtype)
    hcl_m2 = hcl.asarray(np.random.randint(10, size=(k, n)), dtype=dtype)
    hcl_m3 = hcl.asarray(np.zeros((m, n)), dtype=dtype)

    f = hcl.build(s, target)
    f(hcl_m1, hcl_m2, hcl_m3)

    report = f.report()
    latency = report["latency"]["out_matrix"]
gemm()
