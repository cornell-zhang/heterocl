import heterocl as hcl

def test_schedule_pipeline():
  initiation_interval = 3
  a = hcl.placeholder((100,))
  b = hcl.placeholder((100,))
  c = hcl.compute(a.shape, lambda i: a[i] + b[i])
  s = hcl.create_schedule(c)
  s[c].pipeline(c.op.axis[0], initiation_interval)
  ir = hcl.lower(s, [a, b, c])
  pipeline_hint_str = "\"initiation_interval\"="+str(initiation_interval)
  assert pipeline_hint_str in str(ir)

if __name__ == '__main__':
  test_schedule_pipeline()
