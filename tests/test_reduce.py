import heterocl as hcl
import numpy as np


def _test_maxpool(in_shape, out_shape, stride, kernel, test_dtype):
    def max_pool(data):
        h = hcl.reduce_axis(0, kernel)
        w = hcl.reduce_axis(0, kernel)
        return hcl.compute(
            out_shape,
            lambda hh, ww: hcl.max(
                data[stride * hh + h, stride * ww + w], axis=[h, w], dtype=test_dtype
            ),
            name="max_pool",
            dtype=test_dtype,
        )

    A = hcl.placeholder(in_shape, "A", dtype=test_dtype)
    s = hcl.create_schedule([A], max_pool)
    f = hcl.build(s)
    np_a = np.random.randint(0, 10, size=in_shape)
    a = hcl.asarray(np_a, dtype=test_dtype)
    b = hcl.asarray(np.zeros(out_shape), dtype=test_dtype)
    f(a, b)
    np_b = b.asnumpy()
    b_golden = np.zeros(out_shape)
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            b_golden[i, j] = np.max(
                np_a[i * stride : i * stride + kernel, j * stride : j * stride + kernel]
            )
    assert np.allclose(np_b, b_golden)


def test_maxpool():
    in_shape = (2, 8)
    out_shape = (1, 4)
    stride = 2
    kernel = 2

    test_dtypes = [hcl.Int(10), hcl.Int(8), hcl.Int(6), hcl.Fixed(12, 6)]

    for test_dtype in test_dtypes:
        _test_maxpool(in_shape, out_shape, stride, kernel, test_dtype)


def _test_meanpool(in_shape, out_shape, stride, kernel, test_dtype):
    def mean_pool(data):
        h = hcl.reduce_axis(0, kernel)
        w = hcl.reduce_axis(0, kernel)
        return hcl.compute(
            out_shape,
            lambda hh, ww: (hcl.sum(
                data[stride * hh + h, stride * ww + w], axis=[h, w], dtype=test_dtype
            ))/(kernel*kernel),
            name="mean_pool",
            dtype=test_dtype,
        )

    # 0,0 0,1 0,2 0,3 0,4 0,5 0,6 0,7
    # 1,0 1,1 1,2 1,3 1,4 1,5 1,6 1,7
    
    A = hcl.placeholder(in_shape, "A", dtype=test_dtype) # (2,8)
    B = hcl.placeholder(out_shape, "B", dtype=test_dtype) # (1,4)
    s = hcl.create_schedule([A], mean_pool)
    f = hcl.build(s)
    
    np_a = np.random.randint(0, 10, size=in_shape) # random through 0-10
    print(np_a)

    a = hcl.asarray(np_a, dtype=test_dtype) # input array np_a
    b = hcl.asarray(np.zeros(out_shape), dtype=test_dtype) # output array
    f(a, b)
    np_b = b.asnumpy() # turning output array into array
    b_golden = np.zeros(out_shape) # 
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            b_golden[i, j] = np.mean(
                np_a[i * stride : i * stride + kernel, j * stride : j * stride + kernel]
            )
    assert np.allclose(np_b, b_golden)

def test_meanpool():
    in_shape = (2, 8)
    out_shape = (1, 4)
    stride = 2
    kernel = 2

    test_dtypes = [hcl.Float(32)]

    for test_dtype in test_dtypes:
        _test_meanpool(in_shape, out_shape, stride, kernel, test_dtype)