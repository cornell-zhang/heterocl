# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import os
import sys
import numpy as np


def test_stages():
    A = hcl.placeholder((32, 32), "A")
    # C = hcl.placeholder((32, 32), "C")

    def kernel(A):
        B = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "B")
        C = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "C")
        D = hcl.compute(A.shape, lambda i, j: B[i, j] + 1, "D")
        E = hcl.compute(A.shape, lambda i, j: C[i, j] + 1, "E")
        F = hcl.compute(A.shape, lambda i, j: D[i, j] + E[i, j], "F")
        return F

    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="csyn", project="stages-depth-1-new.prj")
    s = hcl.create_schedule([A], kernel)
    s.to(A, target.xcel)
    s.to(kernel.B, s[kernel.D], fifo_depth=1)
    s.to(kernel.C, s[kernel.E], fifo_depth=1)
    s.to(kernel.D, s[kernel.F], fifo_depth=1)
    s.to(kernel.E, s[kernel.F], fifo_depth=1)
    s.to(kernel.F, target.host)
    mod = hcl.build(s, target=target)
    print(mod.src)
    mod()
    # np_A = np.zeros((32, 32))
    # np_C = np.zeros((32, 32))
    # np_F = np.zeros((32, 32))
    # hcl_A = hcl.asarray(np_A)
    # hcl_C = hcl.asarray(np_C)
    # hcl_F = hcl.asarray(np_F)
    # mod(hcl_A, hcl_C)
    # report = mod.report()
    # report.display()


def test_outline():
    A = hcl.placeholder((32, 32), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "B")
        C = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "C")
        D = hcl.compute(A.shape, lambda i, j: B[i, j] + C[i, j], "D")
        return D

    target = hcl.Platform.xilinx_zc706
    target.config(compiler="vivado_hls", mode="debug", project="stages-outline.prj")
    s = hcl.create_schedule([A], kernel)
    s[kernel.B].pipeline(kernel.B.axis[1])
    # s.partition(kernel.B, dim=2)
    func_B = s[kernel.B].outline()
    func_C = s[kernel.C].outline(merge=func_B)
    func_D = s[kernel.D].outline()
    print(s.device_module)
    # func_B_C, func_D = s.outline([s[kernel.B], s[kernel.C]], [s[kernel.D]])
    # func_B, func_C_D = s.outline([s[kernel.B]], [s[kernel.C], s[kernel.D]])
    print(hcl.lower(s))

    # mod = hcl.build(s, top=[func_B_C, func_D], target=target)
    # mod()


def test_outline_extension():
    A = hcl.placeholder((32, 32), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "B")
        C = hcl.compute(A.shape, lambda i, j: B[i, j] + 1, "C")
        D = hcl.compute(A.shape, lambda i, j: C[i, j] + 1, "D")
        return D

    s = hcl.create_schedule([A], kernel)
    # func_B = s[kernel.B].outline()
    # func_C = s[kernel.C].outline(merge=func_B)
    # func_D = s[kernel.D].outline(merge=func_B)
    s.outline(s[kernel.B], s[kernel.C], s[kernel.D], unify=True)
    print(hcl.lower(s))


def test_outline_extension_axis():
    A = hcl.placeholder((32, 32), "A")

    def kernel(A):
        B = hcl.compute((32, 32), lambda i, j: A[i, j] + 1, "B")
        return B

    s = hcl.create_schedule([A], kernel)
    s_B = kernel.B
    x_o, x_i, y_o, y_i = s[s_B].tile(s_B.axis[0], s_B.axis[1], 8, 8)
    func_B = s[s_B].outline(axis=y_i)
    print(s.device_module)
    print(hcl.lower(s))


def test_outline_extension_unify():
    A = hcl.placeholder((32, 32), "A")

    def kernel(A):
        B = hcl.compute((32, 32), lambda i, j: A[i, j] + 1, "B")
        C = hcl.compute((16, 16), lambda i, j: B[i, j] + 1, "C")
        D = hcl.compute((8, 8), lambda i, j: C[i, j] + 1, "D")
        return D

    s = hcl.create_schedule([A], kernel)
    s.outline(s[kernel.B], s[kernel.C], s[kernel.D], unify=True)
    # print(hcl.lower(s))
    mod = hcl.build(s)
    print(s.device_module)
    np_A, np_D = np.zeros((32, 32)), np.zeros((8, 8))
    hcl_A = hcl.asarray(np_A)
    hcl_D = hcl.asarray(np_D)
    mod(hcl_A, hcl_D)
    print(hcl_D.asnumpy())


def test_outline_cpu():
    A = hcl.placeholder((32, 32), "A")

    def kernel(A):
        B = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "B")
        C = hcl.compute(A.shape, lambda i, j: A[i, j] + 1, "C")
        D = hcl.compute(A.shape, lambda i, j: B[i, j] + C[i, j], "D")
        return D

    s = hcl.create_schedule([A], kernel)
    func_B_C, func_D = s.outline([s[kernel.B], s[kernel.C]], [s[kernel.D]])

    mod = hcl.build(s, top=[func_B_C, func_D], target=None)
    np_A, np_B, np_C, np_D = [np.zeros((32, 32))] * 4
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_C = hcl.asarray(np_C)
    hcl_D = hcl.asarray(np_D)
    mod.modules[0](hcl_A, hcl_B, hcl_C)
    mod.modules[1](hcl_B, hcl_C, hcl_D)
    print(hcl_D.asnumpy())


def test_module_mixed_paradigm():
    hcl.init()

    def algorithm(a, b, c):
        @hcl.def_([a.shape, b.shape, c.shape])
        def add(a, b, c):
            with hcl.for_(0, 10, tag="A") as i:
                a[i] = 0
            d = hcl.compute(a.shape, lambda *x: a[x] + b[x], "D")
            hcl.update(c, lambda *x: d[x] + 1, "C")

        add(a, b, c)

    a = hcl.placeholder((10,))
    b = hcl.placeholder((10,))
    c = hcl.placeholder((10,))

    s = hcl.create_schedule([a, b, c], algorithm)
    s.outline([s[algorithm.A], s[algorithm.D], s[algorithm.C]])
    # s[algorithm.A].outline()
    # s[algorithm.D].outline()
    # s[algorithm.C].outline()
    f = hcl.build(s)
    print(s.device_module)

    a = np.random.randint(100, size=(10,))
    b = np.random.randint(100, size=(10,))
    c = np.zeros(10)
    _a = hcl.asarray(a)
    _b = hcl.asarray(b)
    _c = hcl.asarray(c)

    f(_a, _b, _c)

    assert np.array_equal(_c.asnumpy(), b + 1)


if __name__ == "__main__":
    # test_stages()
    # test_outline()
    # test_outline_extension()
    test_outline_extension_unify()
    # test_outline_extension_axis()
    # test_outline_cpu()
    # test_module_mixed_paradigm()
