import unittest

import sys
import json
import heterocl as hcl
import tvm

def basic_func_gen():
  """
  Cover:
    * Kernel pragma
    * Scalar variable declaration
    * Tensor variable declaration
    * Compute function with primary operators
  """
  a = hcl.var("a")
  A = hcl.placeholder((10, 10), "A")
  B = hcl.compute(A.shape, lambda x, y: A[x, y] * a, "B")

  s = hcl.create_schedule(B)
  return hcl.build(s, [a, A, B], target='merlinc')

def loop_sch_func_gen(sch):
  a = hcl.var("a")
  A = hcl.placeholder((10, 10), "A")
  B = hcl.compute(A.shape, lambda x, y: A[x, y] * a, "B")

  s = hcl.create_schedule(B)
  if sch == "parallel":
      s[B].parallel(B.axis[0])
  elif sch == "unroll":
      s[B].unroll(B.axis[0])
  if sch == "pipeline":
      s[B].pipeline(B.axis[0])

  return hcl.build(s, [a, A, B], target='merlinc')

class TestMerlinC(unittest.TestCase):

  def test_basic(self):
    code = basic_func_gen()
    print code
    if UPDATE:
        DB['basic'] = code
    else:
        self.assertEqual(code, DB['basic'])

  def test_loop_schedule(self):
    if UPDATE:
        DB['loop_sch'] = {}

    code = loop_sch_func_gen("parallel")
    print code
    if UPDATE:
        DB['loop_sch']['parallel'] = code
    else:
        self.assertEqual(code, DB['loop_sch']['parallel'])
   
    code = loop_sch_func_gen("unroll")
    print code
    if UPDATE:
        DB['loop_sch']['unroll'] = code
    else:
        self.assertEqual(code, DB['loop_sch']['unroll'])
   
    code = loop_sch_func_gen("pipeline")
    print code
    if UPDATE:
        DB['loop_sch']['pipeline'] = code
    else:
        self.assertEqual(code, DB['loop_sch']['pipeline'])

  def test_downsize(self):
      # FIXME
      self.assertEqual(1, 1)
 
if __name__ == '__main__':
  UPDATE = False
  DB = {}
  if len(sys.argv) > 1 and sys.argv[1] == 'update':
    UPDATE = True
    sys.argv.pop()
  else:
    with open('ref_merlinc.json', 'r') as filep:
        DB = json.load(filep)

  unittest.TextTestRunner().run(
    unittest.TestLoader().loadTestsFromTestCase(TestMerlinC))

  if UPDATE:
    print 'WARNING: Updating test reference, not test is performed'
    with open('ref_merlinc.json', 'w') as filep:
        filep.write(json.dumps(DB, indent=2, separators=(',', ':')))
