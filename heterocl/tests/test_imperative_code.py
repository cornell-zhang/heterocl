import heterocl as hcl

def test_schedule():

  def popcount(A, B): # each element in A is a 32-bit integer
    with hcl.for_(0, A.shape[0], name="x") as x:
      with hcl.for_(0, A.shape[1], name="y") as y:
        B[x, y] = 0
        with hcl.for_(0, 32) as i:
          B[x, y] += A[x, y][i]

  A = hcl.placeholder((10, 20))
  B = hcl.placeholder(A.shape)
  with hcl.stage() as C:
    popcount(A, B)

  def _test_unroll():
    s = hcl.create_schedule(C)
    s[C].unroll(C.x, factor=3)
    ir = hcl.lower(s, [A, B])
    assert "unrolled \"factor\"=3" in str(ir)

  def _test_reorder():
    s = hcl.create_schedule(C)
    s[C].reorder(C.y, C.x)
    ir = hcl.lower(s, [A, B])
    assert str(ir.body.body.body.body).startswith("for (y, 0, 20)")
    assert str(ir.body.body.body.body.body).startswith("for (x, 0, 10)")

  def _test_fuse():
    s = hcl.create_schedule(C)
    s[C].fuse(C.x, C.y)
    ir = hcl.lower(s, [A, B])
    assert str(ir.body.body.body.body).startswith("for (x.y.fused, 0, 200)")

  def _test_split():
    s = hcl.create_schedule(C)
    s[C].split(C.x, factor=3)
    ir = hcl.lower(s, [A, B])
    assert str(ir.body.body.body.body).startswith("for (x.outer, 0, 4)")
    assert str(ir.body.body.body.body.body).startswith("for (x.inner, 0, 3)")
    assert str(ir.body.body.body.body.body.body).startswith(
      "if (((x.outer*3) < (10 - x.inner)))")
    assert str(ir.body.body.body.body.body.body.then_case).startswith(
      "for (y, 0, 20)")

  _test_unroll()
  _test_reorder()
  _test_fuse()
  _test_split()


if __name__ == '__main__':
  test_schedule()
