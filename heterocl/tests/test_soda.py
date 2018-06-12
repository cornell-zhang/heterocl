import unittest

import heterocl as hcl
import tvm

def blur_func_gen():
    img_i = hcl.placeholder((2335, 235), dtype=hcl.UInt(16))
    def blur_x(y, x):
        return (img_i[y, x-1] + img_i[y, x] + img_i[y, x+1])/3
    def blur_y(y, x):
        return (img_t[y-1, x] + img_t[y, x] + img_t[y+1, x])/3
    img_t = hcl.compute((2335, 233), [img_i], blur_x, dtype=hcl.UInt(16))
    img_o = hcl.compute((2333, 233), [img_t], blur_y, dtype=hcl.UInt(16))
    hcl_schedule = hcl.create_schedule(img_o)
    return hcl.build(hcl_schedule, [img_i, img_o], target='soda')

class TestSODA(unittest.TestCase):

    def test_blur(self):
        blur_func = blur_func_gen()
        self.assertEqual(blur_func, 
'''\
kernel: default_function
burst width: 512
dram separate: no
dram bank: 1
unroll factor: 8
border: ignore
cluster: none
iterate: 1
output uint16: compute1(0, 0) = (((compute0(0, -1) + compute0(0, 0)) + compute0(0, 1)) / 3L)
local uint16: compute0(0, 0) = (((placeholder0(-1, 0) + placeholder0(0, 0)) + placeholder0(1, 0)) / 3L)
input uint16: placeholder0(233,)
''')

if __name__ == '__main__':
    unittest.main()

