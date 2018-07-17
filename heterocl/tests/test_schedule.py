import heterocl as hcl

def test_schedule_pipeline():
  initiation_interval = 3
  a = hcl.placeholder((100,))
  b = hcl.placeholder((100,))
  c = hcl.compute(a.shape, lambda i: a[i] + b[i])
  s = hcl.create_schedule(c)
  s[c].pipeline(c.axis[0], initiation_interval)
  ir = hcl.lower(s, [a, b, c])
  pipeline_hint_str = "\"initiation_interval\"="+str(initiation_interval)
  assert pipeline_hint_str in str(ir)

def test_schedule_fuse_loops():
  a = hcl.placeholder((10, 10))
  b = hcl.placeholder((10, 10))
  c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j])
  s = hcl.create_schedule(c)
  s[c].fuse(c.axis[0], c.axis[1])
  ir = hcl.lower(s, [a, b, c])
  assert "i.j.fused" in str(ir)

def test_schedule_reorder_loops():
  a = hcl.placeholder((10, 20, 30))
  b = hcl.placeholder((10, 20, 30))
  c = hcl.compute(a.shape, lambda i, j, k: a[i, j, k] + b[i, j, k])
  s = hcl.create_schedule(c)
  s[c].reorder(c.axis[2], c.axis[1], c.axis[0])
  ir = hcl.lower(s, [a, b, c])
  assert str(ir.body.body).startswith("for (k, 0, 30)")
  assert str(ir.body.body.body).startswith("for (j, 0, 20)")
  assert str(ir.body.body.body.body).startswith("for (i, 0, 10)")


if __name__ == '__main__':
  test_schedule_pipeline()
  test_schedule_fuse_loops()
  test_schedule_reorder_loops()
