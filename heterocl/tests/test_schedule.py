import heterocl as hcl
import tvm

def test_schedule_pipeline():
  initiation_interval = 2
  a = hcl.placeholder((10,))
  b = hcl.placeholder((10,))
  c = hcl.compute(a.shape, lambda i: a[i] + b[i])
  s = hcl.create_schedule(c)
  s[c].pipeline(c.axis[0], initiation_interval)
  ir = hcl.lower(s, [a, b, c])
  pipeline_hint_str = "\"initiation_interval\"="+str(initiation_interval)
  assert pipeline_hint_str in str(ir)

def test_schedule_unroll():
  factor = 2
  a = hcl.placeholder((10,))
  b = hcl.placeholder((10,))
  c = hcl.compute(a.shape, lambda i: a[i] + b[i])
  s = hcl.create_schedule(c)
  s[c].unroll(c.axis[0], factor=factor)
  ir = hcl.lower(s, [a, b, c])
  unroll_hint_str = "\"factor\"="+str(factor)
  assert unroll_hint_str in str(ir)

def test_schedule_fuse_loops():
  a = hcl.placeholder((10, 20, 30, 40))
  b = hcl.placeholder((10, 20, 30, 40))
  c = hcl.compute(a.shape, lambda i, j, k, l: a[i, j, k, l] + b[i, j, k, l])
  s = hcl.create_schedule(c)
  s[c].fuse(c.axis[1], c.axis[2])
  ir = hcl.lower(s, [a, b, c])
  assert "j.k.fused" in str(ir)

def test_schedule_reorder_loops():
  a = hcl.placeholder((10, 20, 30, 40))
  b = hcl.placeholder((10, 20, 30, 40))
  c = hcl.compute(a.shape, lambda i, j, k, l: a[i, j, k, l] + b[i, j, k, l])
  s = hcl.create_schedule(c)
  s[c].reorder(c.axis[2], c.axis[1])
  ir = hcl.lower(s, [a, b, c])
  assert str(ir.body.body).startswith("for (i, 0, 10)")
  assert str(ir.body.body.body).startswith("for (k, 0, 30)")
  assert str(ir.body.body.body.body).startswith("for (j, 0, 20)")
  assert str(ir.body.body.body.body.body).startswith("for (l, 0, 40)")

def test_schedule_compute_at():
  a = hcl.placeholder((10, 20))
  b = hcl.placeholder((10, 20))
  c = hcl.compute(a.shape, lambda i, j: a[i, j] + b[i, j])
  d = hcl.compute(c.shape, lambda i, j: c[i, j])
  s = hcl.create_schedule(d)
  s[c].compute_at(s[d], d.axis[1])
  ir = hcl.lower(s, [a, b, d])
  print ir

def test_schedule_compute_at_tvm():
  a = tvm.placeholder((10, 20))
  b = tvm.placeholder((10, 20))
  c = tvm.compute(a.shape, lambda i, j: a[i, j] + b[i, j])
  d = tvm.compute(c.shape, lambda i, j: c[i, j])
  s = tvm.create_schedule(d.op)
  s[c].compute_at(s[d], d.op.axis[1])
  ir = tvm.lower(s, [a, b, d], simple_mode=True)
  print ir

if __name__ == '__main__':
  # test_schedule_pipeline()
  # test_schedule_unroll()
  # test_schedule_fuse_loops()
  # test_schedule_reorder_loops()
  test_schedule_compute_at()
  test_schedule_compute_at_tvm()
