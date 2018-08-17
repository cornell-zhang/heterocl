import heterocl as hcl
import numpy as np


def test_compute_at():
  A = hcl.placeholder((10, 20, 30), name="A")
  B = hcl.compute(A.shape, lambda i, j, m: A[i, j, m] * 2, name="B")
  C = hcl.compute(B.shape, lambda ii, jj, mm: B[ii, jj, mm] + 1, name="C")

  def _verify_build(sch):
    f = hcl.build(sch, [A, C])
    a_np = np.random.randint(low=0, high=100, size=A.shape)
    a_hcl = hcl.asarray(a_np)
    c_hcl = hcl.asarray(np.zeros(C.shape), dtype="int32")
    f(a_hcl, c_hcl)
    c_np = a_np * 2 + 1
    np.testing.assert_allclose(c_np, c_hcl.asnumpy())

  def test_case_1():
    s0 = hcl.create_schedule(C)
    s0[B].compute_at(s0[C], C.axis[0])
    ir = hcl.lower(s0, [A, C])
    print ir
    _verify_build(s0)
    s1 = hcl.create_schedule(C)
    s1[B].compute_at(s1[C], C.axis[1])
    ir = hcl.lower(s1, [A, C])
    print ir
    _verify_build(s1)
    s2 = hcl.create_schedule(C)
    s2[B].compute_at(s2[C], C.axis[2])
    ir = hcl.lower(s2, [A, C])
    print ir
    _verify_build(s2)

  def test_case_2():
    s = hcl.create_schedule(C)
    s[B].compute_at(s[C], C.axis[2])
    s[C].fuse(C.axis[0], C.axis[1])
    ir = hcl.lower(s, [A, C])
    print ir
    _verify_build(s)

  def test_case_3():
    s = hcl.create_schedule(C)
    s[B].compute_at(s[C], C.axis[2])
    s[C].split(C.axis[0], factor=3)
    s[C].split(C.axis[1], factor=3)
    ir = hcl.lower(s, [A, C])
    print ir
    _verify_build(s)

  def test_case_4():
    s = hcl.create_schedule(C)
    s[B].compute_at(s[C], C.axis[2])
    s[C].reorder(C.axis[1], C.axis[0])
    ir = hcl.lower(s, [A, C])
    print ir
    _verify_build(s)

  def test_case_5():
    s = hcl.create_schedule(C)
    s[B].compute_at(s[C], C.axis[2])
    yo, yi = s[C].split(C.axis[0], factor=3)
    xo, xi = s[C].split(C.axis[1], factor=3)
    s[C].reorder(yo, xo, yi, xi)
    ir = hcl.lower(s, [A, C])
    print ir
    _verify_build(s)

  test_case_1()
  test_case_2()
  test_case_3()
  test_case_4()
  test_case_5()


if __name__ == '__main__':
  test_compute_at()
