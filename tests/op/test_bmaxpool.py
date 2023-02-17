# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import heterocl as hcl
import heterocl.op.bnn as bnn
import numpy as np


def test_binary_maxpool_nchw():
    in_shape = (1, 3, 16, 16)
    out_shape = (1, 3, 8, 8)

    def test_program(A):
        res = bnn.max_pool2d_nchw(A, pooling=[2, 2], stride=[2, 2], padding=[0, 0])
        return res

    A = hcl.placeholder(in_shape, "A", dtype=hcl.UInt(1))
    s = hcl.create_schedule([A], test_program)
    f = hcl.build(s)

    np_A = np.random.randint(0, 2, size=in_shape).astype(np.uint8)
    hcl_A = hcl.asarray(np_A, hcl.UInt(1))
    hcl_res = hcl.asarray(np.zeros(out_shape), hcl.UInt(1))
    f(hcl_A, hcl_res)
    np_res = hcl_res.asnumpy()

    # reference implementation
    golden = np.zeros(out_shape)
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            for k in range(out_shape[2]):
                for l in range(out_shape[3]):
                    golden[i, j, k, l] = np.max(
                        np_A[i, j, k * 2 : k * 2 + 2, l * 2 : l * 2 + 2]
                    )
    assert np.allclose(np_res, golden)


def test_packed_binary_maxpool_nhwc():
    in_shape = (1, 16, 16, 8)
    out_shape = (1, 8, 8, 8)
    bitwidth = 8

    def test_program(A):
        packed_A = hcl.pack(
            A, axis=3, factor=bitwidth, name="packed_A", dtype=hcl.UInt(bitwidth)
        )
        res = bnn.packed_max_pool2d_nhwc(
            packed_A, pooling=[2, 2], stride=[2, 2], padding=[0, 0]
        )
        res_unpacked = hcl.unpack(
            res, axis=3, factor=bitwidth, name="res_unpacked", dtype=hcl.UInt(1)
        )
        return res_unpacked

    A = hcl.placeholder(in_shape, "A", dtype=hcl.UInt(1))
    s = hcl.create_schedule([A], test_program)
    f = hcl.build(s)

    np_A = np.random.randint(0, 2, size=in_shape)
    hcl_A = hcl.asarray(np_A, hcl.UInt(1))
    hcl_res = hcl.asarray(np.zeros(out_shape), hcl.UInt(1))
    f(hcl_A, hcl_res)
    np_res = hcl_res.asnumpy()

    # reference implementation
    golden = np.zeros(out_shape)
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            for k in range(out_shape[2]):
                for l in range(out_shape[3]):
                    golden[i, j, k, l] = np.max(
                        np_A[i, j * 2 : j * 2 + 2, k * 2 : k * 2 + 2, l]
                    )
    # bitpack the result along 2nd dimension
    assert np.allclose(np_res, golden)
