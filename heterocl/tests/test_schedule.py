import heterocl as hcl

def test_schedule_pipeline():
  initiation_interval = 4
  a = hcl.placeholder((10, 20))
  b = hcl.placeholder((10, 20))
  c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j])
  s = hcl.create_schedule(c)
  s[c].pipeline(c.axis[0], initiation_interval)
  ir = hcl.lower(s, [a, b, c])
  pipeline_hint_str = "\"initiation_interval\"="+str(initiation_interval)
  assert pipeline_hint_str in str(ir)

def test_schedule_unroll():
  factor = 4
  a = hcl.placeholder((10, 20))
  b = hcl.placeholder((10, 20))
  c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j])
  s = hcl.create_schedule(c)
  s[c].unroll(c.axis[0], factor=factor)
  ir = hcl.lower(s, [a, b, c])
  unroll_hint_str = "\"factor\"="+str(factor)
  assert unroll_hint_str in str(ir)

def test_schedule_fuse():
  a = hcl.placeholder((10, 20, 30, 40))
  b = hcl.placeholder((10, 20, 30, 40))
  c = hcl.compute(a.shape, lambda i, j, k, l: a[i, j, k, l] + b[i, j, k, l])
  s = hcl.create_schedule(c)
  s[c].fuse(c.axis[1], c.axis[2])
  ir = hcl.lower(s, [a, b, c])
  assert "j.k.fused" in str(ir)

def test_schedule_reorder():
  a = hcl.placeholder((10, 20, 30, 40), name="a")
  b = hcl.placeholder((10, 20, 30, 40), name="b")
  c = hcl.compute(a.shape, lambda i, j, k, l: a[i, j, k, l] + b[i, j, k, l], name="c")
  s = hcl.create_schedule(c)
  s[c].reorder(c.axis[2], c.axis[1])
  ir = hcl.lower(s, [a, b, c])
  assert str(ir.body.body).startswith("for (i, 0, 10)")
  assert str(ir.body.body.body).startswith("for (k, 0, 30)")
  assert str(ir.body.body.body.body).startswith("for (j, 0, 20)")
  assert str(ir.body.body.body.body.body).startswith("for (l, 0, 40)")

def test_schedule_split():
  a = hcl.placeholder((10, 20), name="a")
  b = hcl.placeholder((10, 20), name="b")
  c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")

  def _test_transform_mode():
    s = hcl.create_schedule(c)
    s[c].split(c.axis[1], factor=3, mode="transform")
    ir = hcl.lower(s, [a, b, c])
    assert str(ir.body.body).startswith("for (i, 0, 10)")
    assert str(ir.body.body.body).startswith("for (j.outer, 0, 7)")
    assert str(ir.body.body.body.body).startswith("for (j.inner, 0, 3)")

  def _test_annotate_mode():
    split_factor = 3
    s = hcl.create_schedule(c)
    s[c].split(c.axis[1], factor=split_factor, mode="annotate")
    split_hint_str = "\"split_factor\"="+str(split_factor)
    ir = hcl.lower(s, [a, b, c])
    assert split_hint_str in str(ir)

  _test_transform_mode()
  _test_annotate_mode()

def test_schedule_split_reorder():
  a = hcl.placeholder((10, 20), name="a")
  b = hcl.placeholder((10, 20), name="b")
  c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")
  s = hcl.create_schedule(c)
  yo, yi = s[c].split(c.axis[0], factor=3, mode="transform")
  xo, xi = s[c].split(c.axis[1], factor=3, mode="transform")
  s[c].reorder(yo, xo, yi, xi)
  ir = hcl.lower(s, [a, b, c])
  assert str(ir.body.body).startswith("for (i.outer, 0, 4)")
  assert str(ir.body.body.body).startswith("for (j.outer, 0, 7)")
  assert str(ir.body.body.body.body).startswith("for (i.inner, 0, 3)")
  assert str(ir.body.body.body.body.body).startswith(
    "if (((i.outer*3) < (10 - i.inner)))")
  assert str(ir.body.body.body.body.body.then_case).startswith(
    "for (j.inner, 0, 3)")
  assert str(ir.body.body.body.body.body.then_case.body).startswith(
    "if (((j.outer*3) < (20 - j.inner)))")


if __name__ == '__main__':
  test_schedule_pipeline()
  test_schedule_unroll()
  test_schedule_fuse()
  test_schedule_reorder()
  test_schedule_split()
  test_schedule_split_reorder()
