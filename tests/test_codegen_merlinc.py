import unittest

import os
import sys
import json
import heterocl as hcl

def basic_func_gen():
    """
    Cover:
        * Kernel pragma
        * Scalar variable declaration
        * Tensor variable declaration
        * Compute function with primary operators
    """
    hcl.init()
    a = hcl.placeholder((), "a")
    A = hcl.placeholder((10, 10), "A")
    B = hcl.compute(A.shape, lambda x, y: A[x, y] * a, "B")

    s = hcl.create_schedule([a, A, B])
    return hcl.build(s, target='merlinc')

def loop_sch_func_gen(sch):
    hcl.init()
    a = hcl.placeholder((), "a")
    A = hcl.placeholder((10, 10), "A")
    B = hcl.compute(A.shape, lambda x, y: A[x, y] * a, "B")

    s = hcl.create_schedule([a, A, B])
    if sch == "parallel":
            s[B].parallel(B.axis[0])
    elif sch == "unroll":
            s[B].unroll(B.axis[0])
    if sch == "pipeline":
            s[B].pipeline(B.axis[0])

    return hcl.build(s, target='merlinc')

class TestMerlinC(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DB = {}
        if not "PYTEST_UPDATE" in os.environ:
            with open(os.path.join(THIS_DIR, 'ref_merlinc.json'), 'r') as filep:
                self.DB = json.load(filep)

    @classmethod
    def tearDownClass(self):
        if "PYTEST_UPDATE" in os.environ:
            print('WARNING: Updating test reference, not test is performed')
            with open('ref_merlinc.json', 'w') as filep:
                filep.write(json.dumps(self.DB, indent=2, separators=(',', ':')))

    def test_basic(self):
        code = basic_func_gen()
        if "PYTEST_UPDATE" in os.environ:
            self.DB['basic'] = code
        else:
            self.assertEqual(code, self.DB['basic'])

    def test_loop_schedule(self):
        if "PYTEST_UPDATE" in os.environ:
                self.DB['loop_sch'] = {}

        code = loop_sch_func_gen("parallel")
        if "PYTEST_UPDATE" in os.environ:
                self.DB['loop_sch']['parallel'] = code
        else:
                self.assertEqual(code, self.DB['loop_sch']['parallel'])

        code = loop_sch_func_gen("unroll")
        if "PYTEST_UPDATE" in os.environ:
                self.DB['loop_sch']['unroll'] = code
        else:
                self.assertEqual(code, self.DB['loop_sch']['unroll'])

        code = loop_sch_func_gen("pipeline")
        if "PYTEST_UPDATE" in os.environ:
                self.DB['loop_sch']['pipeline'] = code
        else:
                self.assertEqual(code, self.DB['loop_sch']['pipeline'])

    def test_downsize(self):
            # FIXME
            self.assertEqual(1, 1)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'update':
        os.environ["PYTEST_UPDATE"] = "true"
        sys.argv.pop()

    unittest.TextTestRunner().run(
        unittest.TestLoader().loadTestsFromTestCase(TestMerlinC))

    os.environ.pop("PYTES_UPDATET", None)
