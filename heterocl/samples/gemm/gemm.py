import heterocl as hcl
import numpy as np

import time

def gemm(m=1024, n=1024, k=1024, dtype="int32"):
  matrix_1 = hcl.placeholder((m, k), dtype=dtype)
  matrix_2 = hcl.placeholder((k, n), dtype=dtype)

  k = hcl.reduce_axis(0, k, 'k')
  out_matrix = hcl.compute((m, n), lambda x, y:
    hcl.sum(matrix_1[x, k] * matrix_2[k, y], axis=k, dtype=dtype), dtype=dtype)

  s = hcl.create_schedule(out_matrix)
  block_size = 8
  y0, y1 = s[out_matrix].split(out_matrix.axis[0], factor=block_size)
  x0, x1 = s[out_matrix].split(out_matrix.axis[1], factor=block_size)
  s[out_matrix].reorder(y0, x0, y1, x1)

  f = hcl.build(s, [matrix_1, matrix_2, out_matrix])
  return f


def time_gemm(dtype, m=1024, n=1024, k=1024):
  f = gemm(m=m, n=n, k=k, dtype=dtype)
  np_1 = np.random.randint(10, size=(m, k))
  np_2 = np.random.randint(10, size=(k, n))
  np_3 = np.matmul(np_1, np_2)

  hcl_m1 = hcl.asarray(np_1, dtype=dtype)
  hcl_m2 = hcl.asarray(np_2, dtype=dtype)
  hcl_m3 = hcl.asarray(np.zeros((m, n)), dtype=dtype)
  f(hcl_m1, hcl_m2, hcl_m3)
  begin = time.time()
  for i in range(10):
    f(hcl_m1, hcl_m2, hcl_m3)
  end = time.time()
  print("dtype is: ", dtype)
  print("average of 10 runs of heterocl takes: {} sec".format((end - begin) / 10))
  np.testing.assert_allclose(hcl_m3.asnumpy(), np_3, rtol=1e-03)


if __name__ == "__main__":
  dtypes = ["int32", "float32", hcl.Fixed(32, 16)]
  for dtype in dtypes:
    time_gemm(dtype)
