import unittest

import os
import heterocl as hcl

def run_gcc_test(code_file):
    os.system('g++ -std=c++11 {0}'.format(code_file))
    if not os.path.exists('./a.out'):
        return -1
    ret = os.system('./a.out')
    os.system('rm ./a.out tmp.cpp')
    return ret

class TestC(unittest.TestCase):

    def test_getbit(self):
        hcl.init()
        host = '''\
int main(int argc, char **argv) {
  unsigned int data[] = {123457, 53735, 85343};
  unsigned int ref[] = {0, 1, 0};
  unsigned int out[3];
  default_function(data, out);
  for (int i = 0; i < 3; ++i) {
    if (ref[i] != out[i])
      return 1;
  }
  return 0;
}
'''
        A = hcl.placeholder((3,), "A", dtype=hcl.UInt(32))
        B = hcl.compute(A.shape, lambda x: A[x][7], "B", dtype=hcl.UInt(32))
        s = hcl.create_schedule([A, B])
        code = hcl.build(s, target='merlinc')
        with open('tmp.cpp', 'w') as f:
            f.write(code)
            f.write('\n')
            f.write(host)
        self.assertEqual(run_gcc_test('tmp.cpp'), 0)

    def test_getslice(self):
        hcl.init()
        host = '''\
int main(int argc, char **argv) {
  unsigned int data[] = {123457, 53735, 85343};
  unsigned int ref[] = {4, 3, 2};
  unsigned int out[3];
  default_function(data, out);
  for (int i = 0; i < 3; ++i) {
    if (ref[i] != out[i])
      return 1;
  }
  return 0;
}
'''
        A = hcl.placeholder((3,), "A", dtype=hcl.UInt(32))
        B = hcl.compute(A.shape, lambda x: A[x][7:10], "B", dtype=hcl.UInt(32))
        s = hcl.create_schedule([A, B])
        code = hcl.build(s, target='merlinc')
        with open('tmp.cpp', 'w') as f:
            f.write(code)
            f.write('\n')
            f.write(host)
        self.assertEqual(run_gcc_test('tmp.cpp'), 0)

    def test_scalar(self):
        hcl.init()
        host = '''\
int main(int argc, char **argv) {
  unsigned int data[] = {1,2,3};
  unsigned int out[3] = {0};
  unsigned int ref[] = {1,1,2};
  default_function(data, out);
  for (int i = 0; i < 3; ++i)
    if (ref[i] != out[i])
        return 1;
  return 0;
}
'''
        def bitcount(v):
            out = hcl.scalar(0, "out", dtype=hcl.UInt(32))
            with hcl.for_(0, 3) as i:
                out[0] += v[i]
            return out[0]

        A = hcl.placeholder((3,), "A", dtype=hcl.UInt(32))
        B = hcl.compute(A.shape, lambda x: bitcount(A[x]), "B",
            dtype=hcl.UInt(32))
        s = hcl.create_schedule([A, B])
        code = hcl.build(s, target='merlinc')
        with open('tmp.cpp', 'w') as f:
            f.write(code)
            f.write('\n')
            f.write(host)
        self.assertEqual(run_gcc_test('tmp.cpp'), 0)

if __name__ == '__main__':
    unittest.main()
