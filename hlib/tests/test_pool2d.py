import heterocl as hcl
import numpy as np
import hlib
from hlib.op.nn import get_pad_tuple
from numpy.lib.stride_tricks import as_strided


def pool2d(A, kernel_size, stride, padding, pool_mode='max', layout="NHWC"):
    # Padding
    if layout == "NHWC":
        A = np.pad(A, ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0)), mode='constant')
        output_shape = (
            A.shape[0],
            (A.shape[1] - kernel_size[0]) // stride[0] + 1,
            (A.shape[2] - kernel_size[1]) // stride[1] + 1,
            A.shape[3])
        kernel_size = (kernel_size[0], kernel_size[1])
        A_w = as_strided(A,
                         shape=output_shape + kernel_size,
                         strides=(A.strides[0],
                                  stride[0] * A.strides[1],
                                  stride[1] * A.strides[2],
                                  A.strides[3]) + (A.strides[1],
                                                   A.strides[2]))
        A_w = np.transpose(A_w, (0, 3, 1, 2, 4, 5))
        A_w = A_w.reshape(A.shape[0], A.shape[3], -1, *kernel_size)
        if pool_mode == 'max':
            A_w = np.transpose(A_w.max(axis=(3, 4)), (0, 2, 1))
        elif pool_mode == 'avg':
            A_w = np.transpose(A_w.mean(axis=(3, 4)), (0, 2, 1))
    elif layout == "NCHW":
        A = np.pad(A, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant')
        output_shape = (
            A.shape[0],
            A.shape[1],
            (A.shape[2] - kernel_size[0]) // stride[0] + 1,
            (A.shape[3] - kernel_size[1]) // stride[1] + 1)
        kernel_size = (kernel_size[0], kernel_size[1])
        A_w = as_strided(A,
                         shape=output_shape + kernel_size,
                         strides=(A.strides[0],
                                  A.strides[1],
                                  stride[0] * A.strides[2],
                                  stride[1] * A.strides[3]) + (A.strides[2], A.strides[3]))
        A_w = A_w.reshape(A.shape[0], A.shape[1], -1, *kernel_size)
        if pool_mode == 'max':
            A_w = A_w.max(axis=(3, 4))
        elif pool_mode == 'avg':
            A_w = A_w.mean(axis=(3, 4))
    return A_w.reshape(output_shape).astype(A.dtype)


def pool_test(in_shape, pooling=[2, 2], stride=[1, 1], padding=[0, 0], mode='max', layout="NHWC"):
    hcl.init()
    A = hcl.placeholder(in_shape)

    def max_pool(A, pooling=pooling, stride=stride, padding=padding, layout=layout):
        return hlib.op.nn.max_pool2d(A, pooling, stride, padding, layout)

    def avg_pool(A, pooling=pooling, stride=stride, padding=padding, layout=layout):
        return hlib.op.nn.avg_pool2d(A, pooling, stride, padding, layout)

    if mode == 'max':
        s = hcl.create_schedule([A], max_pool)
    elif mode == 'avg':
        s = hcl.create_schedule([A], avg_pool)
    f = hcl.build(s)
    Pl_h, Pl_w = pooling
    S_h, S_w = stride
    if layout == "NHWC":
        B, H, W, C = in_shape
    elif layout == "NCHW":
        B, C, H, W = in_shape
    P_t, P_l, P_d, P_r = get_pad_tuple(padding, (Pl_h, Pl_w))
    O_h = (H - Pl_h + P_t + P_d) // S_h + 1
    O_w = (W - Pl_w + P_l + P_r) // S_w + 1
    data = hcl.asarray(np.random.randint(50, size=in_shape))
    if layout == "NHWC":
        out_shape = (B, O_h, O_w, C)
    elif layout == "NCHW":
        out_shape = (B, C, O_h, O_w)
    real_out = pool2d(data.asnumpy(), pooling, stride, padding, mode, layout)
    out = hcl.asarray(np.zeros(out_shape))
    f(data, out)
    return np.squeeze(data.asnumpy()), np.squeeze(out.asnumpy()), np.squeeze(real_out)


def pool_assert(data, out, real_out):
    assert(np.array_equal(out, real_out))


def test_max_pool():
    pool_assert(*pool_test((2, 16, 16, 2), pooling=[2, 2], stride=[1, 1], padding=[0, 0], mode='max', layout="NHWC"))
    pool_assert(*pool_test((1, 16, 16, 1), pooling=[4, 4], stride=[2, 2], padding=[0, 0], mode='max', layout="NHWC"))
    pool_assert(*pool_test((2, 4, 4, 2),   pooling=[2, 2], stride=[1, 1], padding=[1, 1], mode='max', layout="NHWC"))
    pool_assert(*pool_test((1, 16, 16, 1), pooling=[4, 4], stride=[2, 2], padding=[1, 1], mode='max', layout="NHWC"))
    pool_assert(*pool_test((2, 1, 4, 4),   pooling=[2, 2], stride=[1, 1], padding=[0, 0], mode='max', layout="NCHW"))
    pool_assert(*pool_test((1, 1, 16, 16), pooling=[4, 4], stride=[2, 2], padding=[0, 0], mode='max', layout="NCHW"))
    pool_assert(*pool_test((2, 1, 4, 4),   pooling=[2, 2], stride=[1, 1], padding=[1, 1], mode='max', layout="NCHW"))
    pool_assert(*pool_test((1, 1, 16, 16), pooling=[4, 4], stride=[2, 2], padding=[1, 1], mode='max', layout="NCHW"))


def test_global_max_pool():
    pool_assert(*pool_test((2, 1, 4, 4),   pooling=[4, 4],   stride=[1, 1], padding=[0, 0], mode='max', layout="NCHW"))
    pool_assert(*pool_test((1, 1, 16, 16), pooling=[16, 16], stride=[2, 2], padding=[0, 0], mode='max', layout="NCHW"))
    pool_assert(*pool_test((5, 5, 4, 4),   pooling=[4, 4],   stride=[2, 2], padding=[0, 0], mode='max', layout="NCHW"))


def test_avg_pool():
    pool_assert(*pool_test((2, 16, 16, 2), pooling=[2, 2], stride=[1, 1], padding=[0, 0], mode='avg', layout="NHWC"))
    pool_assert(*pool_test((1, 16, 16, 1), pooling=[4, 4], stride=[2, 2], padding=[0, 0], mode='avg', layout="NHWC"))
    pool_assert(*pool_test((2, 16, 16, 2), pooling=[2, 2], stride=[1, 1], padding=[1, 1], mode='avg', layout="NHWC"))
    pool_assert(*pool_test((1, 16, 16, 1), pooling=[4, 4], stride=[2, 2], padding=[1, 1], mode='avg', layout="NHWC"))
    pool_assert(*pool_test((2, 1, 4, 4),   pooling=[2, 2], stride=[1, 1], padding=[0, 0], mode='avg', layout="NCHW"))
    pool_assert(*pool_test((1, 1, 16, 16), pooling=[4, 4], stride=[2, 2], padding=[0, 0], mode='avg', layout="NCHW"))
    pool_assert(*pool_test((2, 1, 4, 4),   pooling=[2, 2], stride=[1, 1], padding=[1, 1], mode='avg', layout="NCHW"))
    pool_assert(*pool_test((1, 1, 16, 16), pooling=[4, 4], stride=[2, 2], padding=[1, 1], mode='avg', layout="NCHW"))
